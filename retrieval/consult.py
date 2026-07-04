# Import Libraries
import difflib

import psycopg2.extensions

# Get Candidates
def get_candidates(
    concern: str,
    price_min: float,
    price_max: float,
    conn: psycopg2.extensions.connection,
    skin_type: str | None = None,
    product_names: list[str] | None = None,
    category: str | None = None,
    brand: str | None = None,
    highlights_keyword: str | None = None,
    online_only: bool | None = None,
    sephora_exclusive: bool | None = None,
    good_rating_only: bool = False):
    functions = concern_to_functions(concern, conn)
    ingredient_function_map = functions_to_ingredients(functions, conn)
    ingredients = list(ingredient_function_map.keys())

    if product_names:
        candidates = lookup_products_by_name(product_names, conn)
    else:
        candidates = strict_filter_products(ingredients, price_min, price_max, conn)

    if not candidates:
        return {
            'candidates': [],
            'satisfied': [],
            'dropped': [],
            'ratio': 0.0,
            'status': 'no_strict_match',
        }

    optional_filters = []
    if category:
        optional_filters.append(('category', lambda c: filter_by_category(c, category)))
    if highlights_keyword:
        optional_filters.append(('highlights', lambda c: filter_by_keyword_with_fallback(c, highlights_keyword, ingredients)))
    if online_only is not None:
        optional_filters.append(('online_only', lambda c: filter_by_bool(c, 'online_only', online_only)))
    if sephora_exclusive is not None:
        optional_filters.append(('sephora_exclusive', lambda c: filter_by_bool(c, 'sephora_exclusive', sephora_exclusive)))
    if brand:
        optional_filters.append(('brand', lambda c: filter_by_brand(c, brand)))
    if good_rating_only and skin_type:
        optional_filters.append(('good_rating', lambda c: filter_by_good_rating(c, skin_type, conn)))

    n = len(optional_filters)
    satisfied = []
    dropped = []

    for name, filter_fn in optional_filters:
        attempt = filter_fn(candidates)
        if attempt:
            candidates = attempt
            satisfied.append(name)
        else:
            dropped.append(name)

    ratio = (len(satisfied) / n) if n > 0 else 1.0

    if n > 0 and ratio < 0.75:
        return {
            'candidates': [],
            'satisfied': satisfied,
            'dropped': dropped,
            'ratio': ratio,
            'status': 'below_threshold',
        }

    scored = score_candidates(candidates, ingredient_function_map, functions)
    return {
        'candidates': scored,
        'satisfied': satisfied,
        'dropped': dropped,
        'ratio': ratio,
        'status': 'ok',
    }

# Concern to Function
def concern_to_functions(concern: str, conn: psycopg2.extensions.connection):
    query = """
        SELECT function
        FROM concerns
        WHERE concern = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (concern,))
        rows = cur.fetchall()

    return [row[0] for row in rows]

# Function to Ingredients
def functions_to_ingredients(functions: list[str], conn: psycopg2.extensions.connection):
    query = """
        SELECT ingredient_name, function1, function2
        FROM ingredients
        WHERE function1 = ANY(%s) OR function2 = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(query, (functions, functions))
        rows = cur.fetchall()

    functions_set = set(functions)
    mapping = {}
    for ingredient_name, f1, f2 in rows:
        served = [f for f in (f1, f2) if f in functions_set]
        mapping[ingredient_name] = served

    return mapping

# Strict Filter Products
def strict_filter_products(
    ingredients: list[str], price_min: float,
    price_max: float, conn: psycopg2.extensions.connection):
    query = """
        SELECT product_id, product_name, brand_name, price_usd, ingredients_matched,
               highlights, online_only, sephora_exclusive, secondary_category, tertiary_category
        FROM products
        WHERE ingredients_matched && %s
          AND price_usd >= %s AND price_usd <= %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (ingredients, price_min, price_max))
        rows = cur.fetchall()

    return [_row_to_candidate(row) for row in rows]

# Lookup Products by Name
def lookup_products_by_name(
    product_names: list[str],
    conn: psycopg2.extensions.connection,
    threshold: float = 0.4):
    query = """
        SELECT product_id, product_name, brand_name, price_usd, ingredients_matched,
               highlights, online_only, sephora_exclusive, secondary_category, tertiary_category
        FROM products
    """
    with conn.cursor() as cur:
        cur.execute(query)
        all_rows = cur.fetchall()

    all_names = [row[1] for row in all_rows]
    name_to_row = {row[1]: row for row in all_rows}

    matched_rows = []
    for query_name in product_names:
        close = difflib.get_close_matches(query_name, all_names, n=1, cutoff=threshold)
        if close:
            matched_rows.append(name_to_row[close[0]])

    return [_row_to_candidate(row) for row in matched_rows]

# Filter by Good Rating
def filter_by_good_rating(
    candidates: list[dict],
    skin_type: str,
    conn: psycopg2.extensions.connection,
    min_rating: float = 4.0):
    product_ids = [c['product_id'] for c in candidates]
    query = """
        SELECT product_id
        FROM ratings
        WHERE product_id = ANY(%s) AND skin_type = %s AND avg_rating >= %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (product_ids, skin_type, min_rating))
        good_ids = {row[0] for row in cur.fetchall()}

    return [c for c in candidates if c['product_id'] in good_ids]

# Row to Candidate
def _row_to_candidate(row):
    return {
        'product_id': row[0],
        'product_name': row[1],
        'brand_name': row[2],
        'price_usd': float(row[3]),
        'ingredients_matched': row[4] or [],
        'highlights': row[5] or [],
        'online_only': row[6],
        'sephora_exclusive': row[7],
        'secondary_category': row[8],
        'tertiary_category': row[9],
    }

# Filter by Category
def filter_by_category(candidates: list[dict], category: str):
    kw = category.lower()
    return [
        c for c in candidates
        if (c['tertiary_category'] and kw in c['tertiary_category'].lower())
        or (c['secondary_category'] and kw in c['secondary_category'].lower())
    ]

# Filter by Keyword with Fallbacks
def filter_by_keyword_with_fallback(
    candidates: list[dict],
    keyword: str,
    matched_ingredients: list[str]):
    kw = keyword.lower()

    by_ingredient = [
        c for c in candidates
        if any(kw in ing.lower() for ing in c['ingredients_matched'])
    ]
    if by_ingredient:
        return by_ingredient

    by_category = filter_by_category(candidates, keyword)
    if by_category:
        return by_category

    by_name = [
        c for c in candidates
        if c['product_name'] and kw in c['product_name'].lower()
    ]
    if by_name:
        return by_name

    by_highlights = [
        c for c in candidates
        if any(kw in h.lower() for h in c['highlights'])
    ]
    return by_highlights

# Filter by Bool
def filter_by_bool(candidates: list[dict], field: str, value: bool):
    return [c for c in candidates if c[field] == value]

# Filter by Brand
def filter_by_brand(candidates: list[dict], brand: str):
    b = brand.lower()
    return [c for c in candidates if c['brand_name'] and c['brand_name'].lower() == b]

# Score Candidates
def score_candidates(
    candidates: list[dict],
    ingredient_function_map: dict[str, list[str]],
    functions: list[str],
    ingredient_weight: float = 0.5,
    concern_weight: float = 0.5):
    all_ingredients = set(ingredient_function_map.keys())
    n_functions = len(set(functions))

    scored = []
    for c in candidates:
        matched = [ing for ing in c['ingredients_matched'] if ing in ingredient_function_map]

        ingredient_match = len(set(matched)) / len(all_ingredients) if all_ingredients else 0.0

        covered_functions = set()
        for ing in matched:
            covered_functions.update(ingredient_function_map.get(ing, []))
        concern_match = len(covered_functions) / n_functions if n_functions else 0.0

        similarity_score = ingredient_weight * ingredient_match + concern_weight * concern_match

        scored.append({
            **c,
            'ingredient_match': ingredient_match,
            'concern_match': concern_match,
            'similarity_score': similarity_score,
        })

    return scored