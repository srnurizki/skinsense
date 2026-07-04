# Import Libraries
import os
import datetime
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Global Config
BATCH_SIZE = 32
SEED = 42

# Skin Type Config
IMG_SIZE_ST = (380, 380)
N_CLASSES_ST = 3
TRAIN_DIR_ST = 'data/cleaned/skin_types/train'
VALID_DIR_ST = 'data/cleaned/skin_types/valid'
CHECKPOINT_ST = 'deeplearning/model_artifact/skin_type.keras'
FROZEN_EPOCHS_ST = 50
FROZEN_LR_ST = 1e-4
FINETUNE_EPOCHS_ST = 10
FINETUNE_LR_ST = 1e-5
FINETUNE_N_LAYERS_ST = 20

# Skin Concern Config
IMG_SIZE_SC = (380, 380)
N_CLASSES_SC = 8
TRAIN_DIR_SC = 'data/cleaned/skin_concerns/train'
VALID_DIR_SC = 'data/cleaned/skin_concerns/valid'
CHECKPOINT_SC = 'deeplearning/model_artifact/skin_concern.keras'
FROZEN_EPOCHS_SC = 50
FROZEN_LR_SC = 1e-4
FINETUNE_EPOCHS_SC = 10
FINETUNE_LR_SC = 1e-5
FINETUNE_N_LAYERS_SC = 20

# Preprocess Skin Type
def _build_preprocess_st():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomSharpness(0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)],
        name='augmentation_st')

    def _preprocess(images, labels):
        return preprocess_input(tf.cast(images, tf.float32)), labels

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR_ST, image_size=IMG_SIZE_ST, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=True, seed=SEED)
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR_ST, image_size=IMG_SIZE_ST, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=False, seed=SEED)

    train_ds = train_ds.map(_preprocess)
    valid_ds = valid_ds.map(_preprocess).prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds

# Preprocess Skin Concern
def _build_preprocess_sc():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)],
        name='augmentation_sc')

    def _preprocess(images, labels):
        return preprocess_input(tf.cast(images, tf.float32)), labels

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR_SC, image_size=IMG_SIZE_SC, batch_size=BATCH_SIZE,
        label_mode='categorical', color_mode='rgb', shuffle=True, seed=SEED)
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR_SC, image_size=IMG_SIZE_SC, batch_size=BATCH_SIZE,
        label_mode='categorical', color_mode='rgb', shuffle=False, seed=SEED)

    train_ds = train_ds.map(_preprocess)
    valid_ds = valid_ds.map(_preprocess).prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds

# Model Skin Type
def _build_model_st():
    base = EfficientNetB3(weights='imagenet', include_top=False,
                          input_shape=(*IMG_SIZE_ST, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE_ST, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(N_CLASSES_ST, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='EfficientNetB3_SkinType')

# Model Skin Concern
def _build_model_sc():
    base = EfficientNetB4(weights='imagenet', include_top=False,
                          input_shape=(*IMG_SIZE_SC, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE_SC, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(N_CLASSES_SC, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='EfficientNetB4_SkinConcern')

# Callbacks
def _get_callbacks(checkpoint_path, patience_early=10, patience_lr=3):
    log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    return [
        EarlyStopping(monitor='val_loss', patience=patience_early,
                      restore_best_weights=True, verbose=2),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr,
                          min_lr=1e-7, verbose=2),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                        save_best_only=True, verbose=2),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

# Unfreeze
def _unfreeze(model, n_layers):
    backbone = model.layers[1]
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    trainable = [l for l in backbone.layers if l.trainable]
    print(f'Unfrozen layers: {len(trainable)}')
    return model

# <<<./ Train Skin Type
def train_skin_type():
    train_ds, valid_ds = _build_preprocess_st()

    model = _build_model_st()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2),
        metrics=['accuracy'])

    model.fit(train_ds, validation_data=valid_ds,
              epochs=FROZEN_EPOCHS_ST, callbacks=_get_callbacks(CHECKPOINT_ST))

    model = _unfreeze(model, FINETUNE_N_LAYERS_ST)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR_ST),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2),
        metrics=['accuracy'])

    model.fit(train_ds, validation_data=valid_ds,
              epochs=FINETUNE_EPOCHS_ST,
              callbacks=_get_callbacks(CHECKPOINT_ST, patience_early=3))

# <<<./ Train Skin Concern
def train_skin_concern():
    train_ds, valid_ds = _build_preprocess_sc()

    model = _build_model_sc()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2),
        metrics=['accuracy'])

    model.fit(train_ds, validation_data=valid_ds,
              epochs=FROZEN_EPOCHS_SC, callbacks=_get_callbacks(CHECKPOINT_SC))

    model = _unfreeze(model, FINETUNE_N_LAYERS_SC)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR_SC),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2),
        metrics=['accuracy'])

    model.fit(train_ds, validation_data=valid_ds,
              epochs=FINETUNE_EPOCHS_SC,
              callbacks=_get_callbacks(CHECKPOINT_SC, patience_early=3))


if __name__ == '__main__':
    train_skin_type()
    train_skin_concern()