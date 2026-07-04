# Import Libraries
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from api.schemas import PredictResponse
from deeplearning.inference import predict_skin_type, predict_skin_concern
from storage.imageStore import upload_image
import uuid

# Router
router = APIRouter()

# Limit Uploaded Files
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}

@router.post('/', response_model=PredictResponse)
async def predict(request: Request):
    content_type = request.headers.get('content-type', '')
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type. Use one of: JPEG/JPG, PNG, or WEBP.')

    image_bytes = await request.body()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail='File size exceed 5MB limit.')

    skin_type_result = predict_skin_type(image_bytes)
    if 'error'in skin_type_result:
        raise HTTPException(
            status_code=422,
            detail=skin_type_result['error'])

    skin_concern_result = predict_skin_concern(image_bytes)

    file_id = uuid.uuid4().hex
    unique_filename = f'{file_id}.jpg'
    image_url = await upload_image(image_bytes, unique_filename)

    return PredictResponse(
        skin_type=skin_type_result['skin_type'],
        skin_type_confidence=skin_type_result['skin_type_confidence'],
        skin_concern=skin_concern_result['skin_concern'],
        skin_concern_confidence=skin_concern_result['skin_concern_confidence'],
        image_url=image_url,
        zone_details=skin_type_result.get('zone_details'))