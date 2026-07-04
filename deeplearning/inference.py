# Import Libraries
import numpy as np
import base64
from PIL import Image
import io
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.applications.efficientnet import preprocess_input
import mlflow
from config.settings import MLFLOW_TRACKING_URI

# Config
IMG_SIZE = (380, 380)
MODEL_PATH_SKIN_TYPE = 'models:/SkinTypeClassifier/Production'
MODEL_PATH_SKIN_CONCERN = 'models:/SkinConcernClassifier/Production'
MEDIAPIPE_MODEL_PATH = 'deeplearning/model_artifact/face_landmarker.task'

SKIN_TYPE_CLASSES = ['dry', 'normal', 'oily']
SKIN_CONCERN_CLASSES = [
    'dark spots', 'inflammatory acne',
    'non inflammatory acne black heads', 'non inflammatory acne white heads',
    'pigmentation', 'pores', 'redness', 'wrinkles']

# State Variables
_model_skin_type = None
_model_skin_concern = None
_face_landmarker = None

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load Model
def load_models():
    global _model_skin_type, _model_skin_concern, _face_landmarker

    _model_skin_type = mlflow.keras.load_model(MODEL_PATH_SKIN_TYPE)
    _model_skin_concern = mlflow.keras.load_model(MODEL_PATH_SKIN_CONCERN)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
        num_faces=1)
    _face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Crop Skin Area
def crop_zones(image_rgb: np.ndarray, landmarks):
    h, w = image_rgb.shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(min(xs) * w)
    y1 = int(min(ys) * h)
    x2 = int(max(xs) * w)
    y2 = int(max(ys) * h)
    bw, bh = x2 - x1, y2 - y1

    zones = {
        'forehead':    (x1, y1,              x2, y1 + int(bh * 0.30)),
        'nose':        (x1, y1 + int(bh * 0.30), x2, y1 + int(bh * 0.60)),
        'chin':        (x1, y1 + int(bh * 0.60), x2, y2),
        'cheek_left':  (x1, y1 + int(bh * 0.30), x1 + int(bw * 0.35), y1 + int(bh * 0.70)),
        'cheek_right': (x2 - int(bw * 0.35), y1 + int(bh * 0.30), x2, y1 + int(bh * 0.70)),
    }

    crops = {}
    for name, (zx1, zy1, zx2, zy2) in zones.items():
        crop = image_rgb[zy1:zy2, zx1:zx2]
        if crop.size == 0:
            continue
        size = min(crop.shape[:2])
        ch, cw = crop.shape[:2]
        crops[name] = crop[(ch - size) // 2:(ch + size) // 2,
                           (cw - size) // 2:(cw + size) // 2]
    return crops

# Log-Probability Aggregation
def aggregate_log_prob(preds: list) -> np.ndarray:
    log_probs = np.sum([np.log(p + 1e-9) for p in preds], axis=0)
    exp_log = np.exp(log_probs - np.max(log_probs))
    return exp_log / exp_log.sum()

# Derive Skin Type
def derive_skin_type(zone_preds: dict):
    t_zone = [zone_preds.get('forehead'), zone_preds.get('nose')]
    u_zone = [zone_preds.get('cheek_left'), zone_preds.get('cheek_right'), zone_preds.get('chin')]
    t_zone = [z for z in t_zone if z]
    u_zone = [z for z in u_zone if z]

    t_oily = sum(z == 'oily' for z in t_zone) / len(t_zone) if t_zone else 0
    u_dry = sum(z in ('dry', 'normal') for z in u_zone) / len(u_zone) if u_zone else 0

    if t_oily >= 0.5 and u_dry >= 0.67:
        return 'combination'

    all_zones = t_zone + u_zone
    counts = {c: all_zones.count(c) for c in set(all_zones)}
    return max(counts, key=counts.get)

# Preprocess Skin Cropped Area
def preprocess_crop(crop: np.ndarray):
    img = tf.image.resize(crop, IMG_SIZE).numpy()
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Predict Skin Type
def predict_skin_type(image_bytes: bytes):
    image_rgb = tf.image.decode_image(image_bytes, channels=3).numpy()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = _face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return {'error': 'Wajah tidak terdeteksi.'}

    crops = crop_zones(image_rgb, results.face_landmarks[0])
    zone_probs = {}
    zone_preds = {}
    zone_details = {}

    for name, crop in crops.items():
        probs = _model_skin_type.predict(preprocess_crop(crop), verbose=0)[0]
        zone_probs[name] = probs

        pred_class = SKIN_TYPE_CLASSES[np.argmax(probs)]
        pred_conf = float(np.max(probs))
        zone_preds[name] = pred_class

        crop_img = Image.fromarray(crop.astype(np.uint8))
        buffered = io.BytesIO()
        crop_img.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode("utf8")

        zone_details[name] = {
            'prediction': pred_class,
            'confidence': pred_conf,
            'image_b64': img_str
        }

    agg = aggregate_log_prob(list(zone_probs.values()))
    skin_type = derive_skin_type(zone_preds)

    return {
        'skin_type': skin_type,
        'skin_type_confidence': float(agg.max()),
        'zone_details': zone_details
    }

# Predict Skin Concern
def predict_skin_concern(image_bytes: bytes):
    image = tf.image.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image.numpy())
    image = np.expand_dims(image, axis=0)

    probs = _model_skin_concern.predict(image, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {
        'skin_concern': SKIN_CONCERN_CLASSES[idx],
        'skin_concern_confidence': float(probs[idx])}