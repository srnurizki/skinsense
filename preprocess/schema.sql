-- CREATE TABLE: PRODUCTS
CREATE TABLE IF NOT EXISTS products(
	product_id TEXT PRIMARY KEY,
	product_name TEXT,
	brand_name TEXT,
	rating REAL,
	reviews INTEGER,
	ingredients TEXT,
	highlights TEXT[],
	ingredients_matched TEXT[],
	price_usd NUMERIC(10, 2),
	online_only BOOLEAN,
	sephora_exclusive BOOLEAN,
	primary_category TEXT,
	secondary_category TEXT,
	tertiary_category TEXT);

-- CREATE INDEX ON MATCHING PRODUCT-INGREDIENTS
CREATE INDEX IF NOT EXISTS idx_products_ingredients ON products USING GIN (ingredients_matched);

-- CREATE TABLE: REVIEWS
CREATE TABLE IF NOT EXISTS reviews (
	review_id BIGSERIAL PRIMARY KEY,
	author_id TEXT,
	rating SMALLINT,
	review_text TEXT,
	review_title TEXT,
	skin_tone TEXT,
	skin_type TEXT,
	product_id TEXT REFERENCES products(product_id));

-- CREATE INDEX ON AUTHOR ID, PRODUCT ID, AND SKIN TYPE
CREATE INDEX IF NOT EXISTS idx_reviews_author_id ON reviews(author_id);
CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews (product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_skin_type ON reviews (skin_type);

-- CREATE TABLE: INGREDIENTS
CREATE TABLE IF NOT EXISTS ingredients (
	ingredient_name TEXT PRIMARY KEY,
	function1 TEXT,
	function2 TEXT,
	warning1 TEXT,
	warning2 TEXT,
	ingredient_origin TEXT);

-- CREATE INDEX ON INGREDIENTS FUNCTION
CREATE INDEX IF NOT EXISTS idx_ingredients_function1 ON ingredients(function1);
CREATE INDEX IF NOT EXISTS idx_ingredients_function2 ON ingredients(function2);

-- CREATE TABLE: CONCERN-FUNCTION LOOKUP
CREATE TABLE IF NOT EXISTS concerns (
	concern TEXT,
	function TEXT,
	function_description TEXT,
	concern_description TEXT,
	PRIMARY KEY (concern, function));

-- CREATE TABLE: RATINGS
CREATE TABLE IF NOT EXISTS ratings (
	product_id TEXT REFERENCES products(product_id),
	skin_type TEXT,
	avg_rating REAL,
	review_count INTEGER,
	positive_count INTEGER,
	PRIMARY KEY (product_id, skin_type));

-- CREATE TABLE: USER FEEDBACK
CREATE TABLE IF NOT EXISTS user_feedback (
    id                     SERIAL PRIMARY KEY,
    image_url              TEXT NOT NULL,
    predicted_skin_type    VARCHAR(20) NOT NULL,
    corrected_skin_type    VARCHAR(20) NOT NULL,
    predicted_skin_concern TEXT NOT NULL,
    corrected_skin_concern TEXT NOT NULL,
    session_id             VARCHAR(100),
    used_for_retrain_st    BOOLEAN DEFAULT FALSE,
    used_for_retrain_sc    BOOLEAN DEFAULT FALSE,
    created_at             TIMESTAMP DEFAULT NOW()
);