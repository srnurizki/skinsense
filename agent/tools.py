# Import Libraries
import logging
import requests
from langchain_core.tools import tool
from typing import Literal

from config.settings import (POOL, SERP_API_KEY, usd_idr_rate)
from retrieval.consult import get_candidates
from ranking.weighing import rank

# Setup Basic Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Concern List
ConcernLiteral = Literal[
    'Xerosis & Transepidermal Water Loss', 'Skin Aging, Photoaging & Wrinkles', 'Hyperpigmentation, Dyschromia & Dullness', 'Acne & Hyperseborrhea',
    'Erythema & Sensitivity', 'Puffiness', 'Seborrheic Dermatitis & Dandruff', 'UV Damage', 'pH Balance & Acid Mantle Protection',
    'Hard Water Chelation', 'Active Solubility & Delivery', 'Corneotherapy Washout Risk', 'Degreasing & Foaming', 'Microbial Stability',
    'Rheological Modification', 'Organoleptic Profile']

# Skin Type List
SkinTypeLiteral = Literal['oily', 'combination', 'dry', 'normal']

# Fetch Product Image
def _fetch_image(product_name: str, brand_name: str):
    query = f'{brand_name} {product_name}'
    url = 'https://serpapi.com/search'
    params = {
        'engine': 'google_images',
        'q': query,
        'api_key': SERP_API_KEY,
        'num': 1}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get('images_results', [])
        return results[0]['original'] if results else None
    except (requests.RequestException, KeyError, IndexError):
        return None

# Fetch Product URL
def _fetch_link(product_name: str, brand_name: str):
    query = f'{brand_name} {product_name} site:sephora.com'
    url = 'https://serpapi.com/search'
    params = {
        'engine': 'google',
        'q': query,
        'api_key': SERP_API_KEY,
        'num': 1}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get('organic_results', [])
        if not results:
            print(f'[DEBUG] Query: {query}')
            print(f'[DEBUG] Full response keys: {list(data.keys())}')
            print(f'[DEBUG] search_information: {data.get("search_information")}')
        return results[0]['link'] if results else None
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f'[DEBUG] Exception: {type(e).__name__}: {e}')
        return None

# Retrieve Recommendation
@tool
def retrieve_recommendation(
        concern: ConcernLiteral, skin_type: SkinTypeLiteral, price_idr: float, price_min_idr: float = 0,
        brand: str | None = None, product_names: list[str] | None = None, category: str | None = None,
        highlights_keyword: str | None = None,
        online_only: bool | None = None, sephora_exclusive: bool | None = None, good_rating_only: bool = False):
    """Retrieve product recommendations based on user's concern, skin type, and budget range in IDR. Supports optional filters:
    brand, specific product names to compare, product category, a free-text keyword to match against ingredients/category/product name/highlights
    (in that fallback order), online availability, Sephora exclusivity, and a good-rating constraint (avg_rating >= 4 for the given skin_type).
    Only set good_rating_only=True if the user wants rating as an explicit constraint, not when they're just asking what the rating is.
    Returns candidates plus which optional filters were satisfied or dropped."""
    rate = usd_idr_rate()
    price_max_usd = price_idr * rate
    price_min_usd = price_min_idr * rate

    query_params = {
        'concern': concern,
        'skin_type': skin_type,
        'price_min_usd': price_min_usd,
        'price_max_usd': price_max_usd,
        'brand': brand,
        'product_names': product_names,
        'category': category,
        'highlights_keyword': highlights_keyword,
        'online_only': online_only,
        'sephora_exclusive': sephora_exclusive,
        'good_rating_only': good_rating_only
    }

    logger.info(f'Query params: {query_params}')

    conn = POOL.getconn()
    try:
        result = get_candidates(
            concern,
            price_min_usd,
            price_max_usd,
            conn,
            skin_type=skin_type,
            product_names=product_names,
            category=category,
            brand=brand,
            highlights_keyword=highlights_keyword,
            online_only=online_only,
            sephora_exclusive=sephora_exclusive,
            good_rating_only=good_rating_only)

        logger.info(f'Result: {result}')

        if not result.get('candidates'):
            return result

        top_n = len(product_names) if product_names else 3
        top = rank(result['candidates'], skin_type, conn, top_n=top_n)

    except Exception as e:
        logger.error(f'Error: {e}')
        return {'candidates': [], 'error': str(e)}
    finally:
        POOL.putconn(conn)

    for product in top:
        product['image_url'] = _fetch_image(product['product_name'], product['brand_name'])
        product['product_url'] = _fetch_link(product['product_name'], product['brand_name'])
        product.pop('ingredients_matched', None)
        product.pop('highlights', None)

    result['candidates'] = top
    return result