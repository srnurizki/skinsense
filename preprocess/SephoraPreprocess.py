# Import Libraries
import pandas as pd
from config.settings import (PRODUCT_DIR, REVIEWS_DIR, PRODUCT_CLEANED_DIR,
                             REVIEWS_CLEANED_DIR, INGREDIENTS_CLEANED_DIR, RATINGS_DIR)
import logging
import ast
from tools.timer import timer
import re

# Instantiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Data
@timer
def load():
    products = pd.read_csv(PRODUCT_DIR)
    reviews = pd.read_csv(REVIEWS_DIR)
    logger.info('(Load) SEPHORA Products and Reviews Loaded')
    logger.info(f'(Load) Products Shape: {[products.shape]} | Reviews Shape: {[reviews.shape]}')
    return products, reviews

# Drop Irrelevant Features
@timer
def drop(products: pd.DataFrame, reviews: pd.DataFrame):
    product_drop_features = ['value_price_usd', 'sale_price_usd', 'variation_desc',
                             'child_count', 'child_max_price', 'child_min_price',
                             'brand_id', 'loves_count', 'size', 'variation_type',
                             'variation_value', 'limited_edition', 'new', 'out_of_stock']
    logger.info(f'(Products) These features will be dropped: {product_drop_features}')
    products = products.drop(product_drop_features, axis=1, errors='ignore')
    reviews_drop_features = ['total_feedback_count', 'total_neg_feedback_count',
                             'total_pos_feedback_count', 'submission_time',
                             'Unnamed: 0', 'is_recommended', 'eye_color', 'hair_color',
                             'helpfulness', 'product_name', 'brand_name', 'price_usd']
    logger.info(f'(Reviews) These features will be dropped: {reviews_drop_features}')
    reviews = reviews.drop(reviews_drop_features, axis=1, errors='ignore')
    return products, reviews

# Filter Skin and Body Care Products
@timer
def filter(products: pd.DataFrame, reviews: pd.DataFrame):
    mask = products['primary_category'].isin(['Skincare', 'Bath & Body']) | (
            (products['primary_category'] == 'Men') & (products['secondary_category'] == 'Skincare'))
    products = products[mask]
    reviews = reviews[reviews['product_id'].isin(products['product_id'])]
    return products, reviews

# Impute NAN
@timer
def impute(products: pd.DataFrame, reviews: pd.DataFrame):
    # Products: df1
    products['rating'] = products['rating'].fillna(0)
    products['reviews'] = products['reviews'].fillna(0)
    impute_products = {
        'ingredients': 'No ingredients listed',
        'highlights': 'No highlights described',
        'secondary_category': 'No secondary category',
        'tertiary_category': 'No tertiary category'}
    products = products.fillna(impute_products)
    logger.info('(Products) Missing values have been handled')

    # Reviews: df2
    impute_reviews = {
        'review_text': 'No reviews written',
        'review_title': 'No reviews written',
        'skin_tone': 'No skin tone described',
        'skin_type': 'No skin type described'}
    reviews = reviews.fillna(impute_reviews)
    logger.info('(Reviews) Missing values have been handled')
    return products, reviews

# Parse List of 'Values'
def parse(value: str):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return str(value)
    if not isinstance(parsed, list):
        return str(parsed)
    cleaned = [item for item in parsed if not item.strip().endswith(':')]
    return ', '.join(cleaned)

# Build Pattern
def patternize(ingredient_name: list):
    uniques = sorted(set(ingredient_name), key=len, reverse=True)
    escaped = [re.escape(name) for name in uniques]
    pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
    lookup = {name.lower(): name for name in ingredient_name}
    return pattern, lookup

# Match Ingredients
def matching(ingredient_names: str, pattern: re.Pattern, lookup: dict):
    matches = pattern.findall(ingredient_names)
    canonical = {lookup.get(match.lower(), match) for match in matches}
    return list(canonical)

# Prepare Products
@timer
def prepare(products: pd.DataFrame, ingredients: pd.DataFrame):
    products['ingredients'] = products['ingredients'].apply(parse)
    products['highlights'] = products['highlights'].apply(parse)
    pattern, lookup = patternize(ingredients['ingredient_name'].tolist())
    products['ingredients_matched'] = products['ingredients'].apply(
        lambda name: matching(name, pattern, lookup))
    return products

# Weighted Rating
@timer
def rating(reviews: pd.DataFrame):
    valid = reviews[reviews['skin_type'] != 'No skin type described']
    grouped = valid.groupby(['product_id', 'skin_type']).agg(
        avg_rating=('rating', 'mean'),
        review_count=('rating', 'count'),
        positive_count=('rating', lambda x: (x >= 4).sum())
    ).reset_index()
    return grouped

# Save to CSV
@timer
def save(products: pd.DataFrame, reviews: pd.DataFrame):
    products.to_csv(PRODUCT_CLEANED_DIR, index=False)
    reviews.to_csv(REVIEWS_CLEANED_DIR, index=False)
    return None

# Pipeline
def main():
    products, reviews = load()
    products, reviews = drop(products, reviews)
    products, reviews = impute(products, reviews)
    products, reviews = filter(products, reviews)
    try:
        ingredients = pd.read_csv(INGREDIENTS_CLEANED_DIR)
    except FileNotFoundError:
        return 'Run IngredientsPreprocess.py first'
    products = prepare(products, ingredients)
    ratings = rating(reviews)
    ratings.to_csv(RATINGS_DIR, index=False)
    save(products, reviews)

# Init
if __name__ == "__main__":
    main()