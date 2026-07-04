# Import Libraries
import psycopg2.extensions

# Rank Candidates
def rank(
    candidates: list[dict],
    skin_type: str,
    conn: psycopg2.extensions.connection,
    top_n: int = 3):
    product_ids = [c['product_id'] for c in candidates]
    ratings = fetch(product_ids, skin_type, conn)
    rating_map = {r['product_id']: r for r in ratings}

    scored = []
    for c in candidates:
        r = rating_map.get(c['product_id'], {'avg_rating': None, 'review_count': 0, 'positive_count': 0})
        scored.append({
            **c,
            'avg_rating': r['avg_rating'],
            'rating_score': wilson_score(r['positive_count'], r['review_count'])
        })

    return sorted(scored, key=lambda x: x['rating_score'], reverse=True)[:top_n]

# Fetch
def fetch(
    product_ids: list[str],
    skin_type: str,
    conn: psycopg2.extensions.connection):
    query = """
        SELECT product_id, avg_rating, review_count, positive_count
        FROM ratings
        WHERE product_id = ANY(%s) AND skin_type = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (product_ids, skin_type))
        rows = cur.fetchall()

    return [
        {
            'product_id': row[0],
            'avg_rating': float(row[1]) if row[1] is not None else None,
            'review_count': row[2],
            'positive_count': row[3],
        }
        for row in rows
    ]

# Calculate Wilson Score
def wilson_score(positive_count: int, review_count: int, z: float = 1.96):
    if review_count == 0:
        return 0.0
    p = positive_count / review_count
    n = review_count
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (center - margin) / denom