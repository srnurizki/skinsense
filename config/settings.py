# Import Libraries
import os
import time
import requests
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool

load_dotenv(override=True)

# PostgresDB
CONNECTION_STRING = os.getenv('CONNECTION_STRING')

# APIs
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')

# GCS
GCS_BUCKET_NAME='staca-user-uploads'
GCS_PROJECT_ID='udemy-learn-492913'

# GitHub
GITHUB_TOKEN=os.getenv('GITHUB_TOKEN')
GITHUB_REPO='srnurizki/skinsense'

# MLflow
MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_DEFAULT_ARTIFACT_ROOT = os.getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT')

# <<<./ Data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PRODUCT_DIR = os.path.join(DATA_DIR, r'raw/sephora/product.csv')
REVIEWS_DIR = os.path.join(DATA_DIR, r'raw/sephora/reviews.csv')
INGREDIENTS_DIR = os.path.join(DATA_DIR, r'raw/ingredients/ingredients.xlsx')
PRODUCT_CLEANED_DIR = os.path.join(DATA_DIR, r'cleaned/sephora/products-cleaned.csv')
REVIEWS_CLEANED_DIR = os.path.join(DATA_DIR, r'cleaned/sephora/reviews-cleaned.csv')
INGREDIENTS_CLEANED_DIR = os.path.join(DATA_DIR, r'cleaned/ingredients/ingredients-cleaned.csv')
CONCERNS_DIR = os.path.join(DATA_DIR, r'cleaned/concerns/concerns.csv')
SKIN_TYPE_TRAIN_DIR = os.path.join(DATA_DIR, r'cleaned/skin_types/train')
SKIN_CONCERN_TRAIN_DIR = os.path.join(DATA_DIR, r'cleaned/skin_concerns/train')
RATINGS_DIR = os.path.join(DATA_DIR, r'cleaned/ratings/ratings.csv')

# Connection Pool
POOL = ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=CONNECTION_STRING)

rate_cache = {'rate': None, 'fetched_at': 0}
CACHE_TTL = 86400

# Convert USD to IDR
def usd_idr_rate():
    now = time.time()
    if rate_cache['rate'] is not None and now - rate_cache['fetched_at'] < CACHE_TTL:
        return rate_cache['rate']

    url = 'https://api.exchangerate.host/live'
    params = {
        'access_key': EXCHANGE_API_KEY,
        'source': 'USD',
        'currencies': 'IDR'}

    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    usd_to_idr = data['quotes']['USDIDR']

    rate_cache['rate'] = 1 / usd_to_idr
    rate_cache['fetched_at'] = now
    return rate_cache['rate']


