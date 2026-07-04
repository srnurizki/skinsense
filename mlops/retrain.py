# Import Libraries
import os
import uuid
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from datetime import datetime
from sklearn.metrics import f1_score
from google.cloud import storage
from config.settings import (DATABASE_URL, GCS_BUCKET_NAME, GCS_PROJECT_ID,
                              SKIN_TYPE_TRAIN_DIR, SKIN_CONCERN_TRAIN_DIR,
                              MLFLOW_TRACKING_URI)
from mlops.dataCollector import (collect_skin_type_samples,
                                   collect_skin_concern_samples,
                                   mark_as_used)
import argparse

# Config
BATCH_SIZE = 32
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-5

SKIN_TYPE_CLASSES = ['dry', 'normal', 'oily']
SKIN_CONCERN_CLASSES = ['dark spots', 'inflammatory acne',
    'non inflammatory acne black heads', 'non inflammatory acne white heads',
    'pigmentation', 'pores', 'redness', 'wrinkles']

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Augmentation
_augmentation_st = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomSharpness(0.1),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)],
    name='augmentation_st')

_augmentation_sc = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)],
    name='augmentation_sc')

# Download Data from GCS
def _download_from_gcs(image_url: str, local_path: str):
    client = storage.Client(project=GCS_PROJECT_ID)
    blob = storage.Blob.from_string(image_url, client=client)
    blob.download_to_filename(local_path)

# Load Image
def _load_image(image_url: str, img_size: tuple) -> np.ndarray:
    local_path = f'/tmp/{uuid.uuid4().hex}.jpg'
    _download_from_gcs(image_url, local_path)
    img = tf.keras.utils.load_img(local_path, target_size=img_size)
    img = tf.keras.utils.img_to_array(img)
    os.remove(local_path)
    return img

# Balance Data
def _oversample_to_balance(images: list, labels: list, augmentation) -> tuple:
    label_arr = np.array(labels)
    image_arr = np.array(images)
    classes = np.unique(label_arr)
    max_count = max(np.sum(label_arr == c) for c in classes)

    aug_images, aug_labels = list(image_arr), list(label_arr)
    for cls in classes:
        cls_images = image_arr[label_arr == cls]
        needed = max_count - len(cls_images)
        for i in range(needed):
            src = cls_images[i % len(cls_images)]
            augmented = augmentation(
                tf.expand_dims(src, 0), training=True)[0].numpy()
            aug_images.append(augmented)
            aug_labels.append(cls)

    return np.array(aug_images), np.array(aug_labels)

# Load Existing Dataset
def _load_existing_dataset(train_dir: str, img_size: tuple) -> tuple:
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=None,
        label_mode='int', color_mode='rgb')
    images = np.array([img.numpy() for img, _ in ds])
    labels = np.array([lbl.numpy() for _, lbl in ds])
    return images, labels

# Get Best F1
def _get_current_best_f1(model_name: str) -> float:
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=['Production'])
        if not versions:
            return 0.0
        run = client.get_run(versions[0].run_id)
        return float(run.data.metrics.get('macro_f1', 0.0))
    except Exception:
        return 0.0

# Finetune
def _finetune(model_name: str, model, feedback: list, classes: list,
              train_dir: str, img_size: tuple, augmentation):
    new_images, new_labels = [], []
    for row in feedback:
        try:
            img = _load_image(row['image_url'], img_size)
            new_images.append(img)
            new_labels.append(classes.index(row['label']))
        except Exception as e:
            print(f'Skip {row["image_url"]}: {e}')

    new_images, new_labels = _oversample_to_balance(new_images, new_labels, augmentation)
    old_images, old_labels = _load_existing_dataset(train_dir, img_size)

    all_images = np.concatenate([old_images, new_images])
    all_labels = np.concatenate([old_labels, new_labels])
    idx = np.random.permutation(len(all_images))
    all_images, all_labels = all_images[idx], all_labels[idx]

    ds_train = tf.data.Dataset.from_tensor_slices(
        (all_images, all_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0),
        metrics=['accuracy'])

    with mlflow.start_run():
        mlflow.log_params({
            'model': model_name,
            'finetune_epochs': FINETUNE_EPOCHS,
            'finetune_lr': FINETUNE_LR,
            'new_samples': len(feedback),
        })

        model.fit(ds_train, epochs=FINETUNE_EPOCHS, verbose=1)

        y_true, y_pred = [], []
        for img_batch, lbl_batch in ds_train:
            preds = model.predict(img_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(lbl_batch.numpy())

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        mlflow.log_metric('macro_f1', macro_f1)
        print(f'{model_name} Macro F1: {macro_f1:.4f}')

        current_best = _get_current_best_f1(model_name)
        if macro_f1 > current_best:
            mlflow.keras.log_model(model, model_name.lower())
            run_id = mlflow.active_run().info.run_id
            version = mlflow.register_model(
                f'runs:/{run_id}/{model_name.lower()}', model_name)
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=version.version, stage='Production')
            print(f'{model_name} promoted (Macro F1: {macro_f1:.4f})')
            mark_as_used([r['image_url'] for r in feedback],
                         'skin_type' if 'Type' in model_name else 'skin_concern')
        else:
            print(f'{model_name} is not improving. Keep current.')

# Retrain
def run_skin_type():
    feedback = collect_skin_type_samples()
    if not feedback:
        print('No skin_type feedback available.')
        return
    model = tf.keras.models.load_model('models:/SkinTypeClassifier/Production')
    img_size = model.input_shape[1:3]
    _finetune('SkinTypeClassifier', model, feedback,
              SKIN_TYPE_CLASSES, SKIN_TYPE_TRAIN_DIR, img_size, _augmentation_st)

def run_skin_concern():
    feedback = collect_skin_concern_samples()
    if not feedback:
        print('No skin_concern feedback available.')
        return
    model = tf.keras.models.load_model('models:/SkinConcernClassifier/Production')
    img_size = model.input_shape[1:3]
    _finetune('SkinConcernClassifier', model, feedback,
              SKIN_CONCERN_CLASSES, SKIN_CONCERN_TRAIN_DIR, img_size, _augmentation_sc)

# Run Retrain
def run():
    run_skin_type()
    run_skin_concern()

# Init
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['skin_type', 'skin_concern'], default=None)
    args = parser.parse_args()

    if args.model == 'skin_type':
        run_skin_type()
    elif args.model == 'skin_concern':
        run_skin_concern()
    else:
        run()
