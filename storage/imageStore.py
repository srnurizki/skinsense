# Import Libraries
import uuid
from google.cloud import storage
from config.settings import GCS_BUCKET_NAME, GCS_PROJECT_ID

# Singleton Client
_client = None
_bucket = None

# Get GCS Bucket
def _get_bucket():
    global _client, _bucket
    if _bucket is None:
        _client = storage.Client(project=GCS_PROJECT_ID)
        _bucket = _client.get_bucket(GCS_BUCKET_NAME)
    return _bucket

# Upload User Uploaded Image to Bucket
async def upload_image(image_bytes: bytes, original_filename: str):
    bucket = _get_bucket()
    ext = original_filename.rsplit('.', 1)[-1] if '.' in original_filename else 'jpg'
    blob_name = f'user-uploads/{uuid.uuid4().hex}.{ext}'
    blob = bucket.blob(blob_name)

    blob.upload_from_string(image_bytes, content_type=f'image/{ext}')
    blob.make_public()

    return blob.public_url