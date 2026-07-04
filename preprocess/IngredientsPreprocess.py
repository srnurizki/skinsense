# Import Libraries
import pandas as pd
from config.settings import INGREDIENTS_DIR, INGREDIENTS_CLEANED_DIR
import logging
from tools.timer import timer

# Instantiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Data
@timer
def load():
    df = pd.read_excel(INGREDIENTS_DIR)
    logger.info('Ingredients Data Loaded')
    logger.info(f'Data Shape: {[df.shape]}')
    return df

# Drop Irrelevant Features
@timer
def drop(df: pd.DataFrame):
    drop_feature = ['ingredient_charge']
    logger.info(f'(Ingredients) These features will be dropped: {drop_feature}')
    df = df.drop(drop_feature, axis=1, errors='ignore')
    return df

# Impute NAN
@timer
def impute(df: pd.DataFrame):
    df['function1'] = df['function1'].str.strip()
    df['function2'] = df['function2'].str.strip()
    df['function2'] = df['function2'].fillna('No other function')
    df['warning1'] = df['warning1'].fillna('No known warning')
    df['warning2'] = df['warning2'].fillna('No known warning')
    logger.info('(Ingredients) Missing values have been handled')
    return df

# Save to CSV
@timer
def save(df: pd.DataFrame):
    df.to_csv(INGREDIENTS_CLEANED_DIR, index=False)
    return None

# Pipeline
def main():
    df = load()
    df = drop(df)
    df = impute(df)
    save(df)

# Init
if __name__ == "__main__":
    main()