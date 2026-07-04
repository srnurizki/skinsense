# Import Libraries
import ast
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from config.settings import (PRODUCT_CLEANED_DIR, REVIEWS_CLEANED_DIR, INGREDIENTS_CLEANED_DIR,
                             RATINGS_DIR, CONCERNS_DIR)
import logging
import os
from tools.timer import timer
from dotenv import load_dotenv

# Instantiate Logger and Dependencies
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
CONNECTION_STRING = os.getenv('CONNECTION_STRING')
schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')

# Create Table
@timer
def create_table(connection, schema=schema_path):
    with open(schema) as f:
        ddl = f.read()
    with connection.cursor() as cursor:
        cursor.execute(ddl)
    connection.commit()
    logger.info('(Postgres) Tables created from schema.sql')

# Load Cleaned Data
@timer
def clean_load():
    products = pd.read_csv(PRODUCT_CLEANED_DIR)
    reviews = pd.read_csv(REVIEWS_CLEANED_DIR)
    ingredients = pd.read_csv(INGREDIENTS_CLEANED_DIR)
    rating = pd.read_csv(RATINGS_DIR)
    concerns = pd.read_csv(CONCERNS_DIR)
    return products, reviews, ingredients, rating, concerns

# Parse List of Strings
def parse(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

# Ingest Ingredients
@timer
def ingest_ingredients(connection, ingredients):
    column = ingredients.columns.tolist()
    rows = list(ingredients[column].itertuples(index=False, name=None))
    query = f'INSERT INTO ingredients ({", ".join(column)}) VALUES %s ON CONFLICT (ingredient_name) DO NOTHING'
    with connection.cursor() as cursor:
        execute_values(cursor, query, rows)
    connection.commit()
    logger.info(f'(Postgres) Ingestion of {len(rows)} rows completed for Ingredients')

# Ingest Products
@timer
def ingest_products(connection, products):
    products = products.copy()
    products['ingredients_matched'] = products['ingredients_matched'].apply(parse)
    products['online_only'] = products['online_only'].astype(bool)
    products['sephora_exclusive'] = products['sephora_exclusive'].astype(bool)
    products['highlights'] = products['highlights'].astype(str).str.split(', ')
    column = products.columns.tolist()
    rows = list(products[column].itertuples(index=False, name=None))
    query = f'INSERT INTO products ({", ".join(column)}) VALUES %s ON CONFLICT (product_id) DO NOTHING'
    with connection.cursor() as cursor:
        execute_values(cursor, query, rows)
    connection.commit()
    logger.info(f'(Postgres) Ingestion of {len(rows)} rows completed for Products')

# Ingest Reviews
@timer
def ingest_reviews(connection, reviews):
    column = reviews.columns.tolist()
    rows = list(reviews[column].itertuples(index=False, name=None))
    query = f'INSERT INTO reviews ({", ".join(column)}) VALUES %s ON CONFLICT (review_id) DO NOTHING'
    with connection.cursor() as cursor:
        execute_values(cursor, query, rows)
    connection.commit()
    logger.info(f'(Postgres) Ingestion of {len(rows)} rows completed for Reviews')

# Ingest Ratings
@timer
def ingest_ratings(connection, ratings):
    column = ratings.columns.tolist()
    rows = list(ratings[column].itertuples(index=False, name=None))
    query = f'INSERT INTO ratings ({", ".join(column)}) VALUES %s ON CONFLICT (product_id, skin_type) DO NOTHING'
    with connection.cursor() as cursor:
        execute_values(cursor, query, rows)
    connection.commit()
    logger.info(f'(Postgres) Ingestion of {len(rows)} rows completed for Ratings')

# Ingest Concerns
@timer
def ingest_concerns(connection, concerns):
    column = concerns.columns.tolist()
    rows = list(concerns[column].itertuples(index=False, name=None))
    query = f'INSERT INTO concerns ({", ".join(column)}) VALUES %s ON CONFLICT (concern, function) DO NOTHING'
    with connection.cursor() as cursor:
        execute_values(cursor, query, rows)
    connection.commit()
    logger.info(f'(Postgres) Ingestion of {len(rows)} rows completed for Concerns')

# Pipeline
def main():
    products, reviews, ingredients, rating, concerns = clean_load()
    connection = psycopg2.connect(CONNECTION_STRING)
    create_table(connection)
    #ingest_ingredients(connection, ingredients)
    #ingest_concerns(connection, concerns)
    #ingest_products(connection, products)
    ingest_reviews(connection, reviews)
    ingest_ratings(connection, rating)
    connection.close()

# Init
if __name__ == '__main__':
    main()