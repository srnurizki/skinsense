# Import Libraries
from psycopg2.extras import RealDictCursor
from config.settings import POOL

# Minimum Sample
MIN_SAMPLES = 50

# Connect
def _query(sql, params=None, fetch=True, commit=False):
    conn = POOL.getconn()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql, params)
        result = cursor.fetchall() if fetch else None
        if commit:
            conn.commit()
        cursor.close()
        return result
    finally:
        POOL.putconn(conn)


# Get Skin Type Feedbacks
def collect_skin_type_samples():
    return _query("""
        SELECT image_url, corrected_skin_type AS label
        FROM user_feedback
        WHERE predicted_skin_type != corrected_skin_type
          AND used_for_retrain_st = FALSE
        ORDER BY created_at ASC;
    """)


# Get Skin Concern Feedbacks
def collect_skin_concern_samples():
    return _query("""
        SELECT image_url, corrected_skin_concern AS label
        FROM user_feedback
        WHERE predicted_skin_concern != corrected_skin_concern
          AND used_for_retrain_sc = FALSE
        ORDER BY created_at ASC;
    """)


# Check Threshold
def has_enough_samples(model_type: str) -> bool:
    if model_type == 'skin_type':
        return len(collect_skin_type_samples()) >= MIN_SAMPLES
    return len(collect_skin_concern_samples()) >= MIN_SAMPLES

# Mark Sample as 'used_for_retrain'
def mark_as_used(image_urls: list, model_type: str):
    col = 'used_for_retrain_st' if model_type == 'skin_type' else 'used_for_retrain_sc'
    _query(
        f'UPDATE user_feedback SET {col} = TRUE WHERE image_url = ANY(%s);',
        (image_urls,), fetch=False, commit=True)
