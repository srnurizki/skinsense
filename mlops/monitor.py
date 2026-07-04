# Import Libraries
import os

import requests
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

from config.settings import POOL, GITHUB_REPO
from mlops.data_collector import has_enough_samples
from dotenv import load_dotenv

# Config
ERROR_RATE_THRESHOLD = 0.2
DRIFT_WINDOW_DAYS    = 30
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Compute Drift
def _compute_drift(predicted_col: str, corrected_col: str) -> float:
    since = datetime.utcnow() - timedelta(days=DRIFT_WINDOW_DAYS)
    query = f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN {predicted_col} != {corrected_col} THEN 1 ELSE 0 END) AS errors
        FROM user_feedback
        WHERE created_at >= %s;
    """
    conn = POOL.getconn()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (since,))
        row = cursor.fetchone()
        cursor.close()
    finally:
        POOL.putconn(conn)

    if not row or row['total'] == 0:
        return 0.0
    return row['errors'] / row['total']


# Trigger GitHub Actions
def _trigger_retrain(event_type: str):
    url = f'https://api.github.com/repos/{GITHUB_REPO}/dispatches'
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github+json',
    }
    response = requests.post(url, headers=headers, json={'event_type': event_type})
    response.raise_for_status()
    print(f'Triggered {event_type}: {response.status_code}')

# Monitor Drift
def _check_and_trigger(model_type: str, predicted_col: str, corrected_col: str):
    drift = _compute_drift(predicted_col, corrected_col)
    print(f'[{model_type}] drift (last {DRIFT_WINDOW_DAYS}d): {drift:.2%}')

    drift_exceeded = drift >= ERROR_RATE_THRESHOLD
    enough_samples = has_enough_samples(model_type)

    if drift_exceeded and enough_samples:
        event_type = f'retrain_{model_type}'
        print(f'[{model_type}] Conditions met. Triggering {event_type}...')
        _trigger_retrain(event_type)
    else:
        print(f'[{model_type}] Not triggered. drift_exceeded={drift_exceeded}, enough_samples={enough_samples}')

# Run Monitor
def run():
    _check_and_trigger('skin_type', 'predicted_skin_type', 'corrected_skin_type')
    _check_and_trigger('skin_concern', 'predicted_skin_concern', 'corrected_skin_concern')

# Init
if __name__ == '__main__':
    run()