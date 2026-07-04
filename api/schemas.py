# Import Libraries
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Predict
class PredictResponse(BaseModel):
    skin_type: str
    skin_type_confidence: float
    skin_concern: str
    skin_concern_confidence: float
    image_url: Optional[str]
    zone_details: Optional[Dict[str, Any]] = None

# Chat
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []
    session_id: Optional[str] = None

# Feedback
class FeedbackRequest(BaseModel):
    image_url: str
    predicted_skin_type: str
    corrected_skin_type: str
    predicted_skin_concern: str
    corrected_skin_concern: str
    session_id: Optional[str] = None