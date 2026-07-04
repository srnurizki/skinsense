# Import Libraries
from fastapi import APIRouter, HTTPException
from api.schemas import FeedbackRequest
from config.settings import CONNECTION_STRING
import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter()

# Connect
def _get_connection():
    return psycopg2.connect(CONNECTION_STRING)

# Insert Feedback
@router.post('/')
def submit_feedback(request: FeedbackRequest):
    query = """
        INSERT INTO user_feedback (
            image_url,
            predicted_skin_type,
            corrected_skin_type,
            predicted_skin_concern,
            corrected_skin_concern,
            session_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    try:
        conn   = _get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (
            request.image_url,
            request.predicted_skin_type,
            request.corrected_skin_type,
            request.predicted_skin_concern,
            request.corrected_skin_concern,
            request.session_id))
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        return {'id': result['id'], 'status': 'ok'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))