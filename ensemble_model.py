import os
import shutil
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Average, BatchNormalization
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet101, VGG16, DenseNet121, EfficientNetB0
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import CosineDecay
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except ModuleNotFoundError:
    os.system('pip install tensorflow')
    import tensorflow as tf

# ---------------------- GPU Konfigürasyonu ----------------------
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
except Exception as e:
    raise RuntimeError("Failed to initialize GPU strategy. Ensure TensorFlow is correctly installed and GPU is available.") from e

<<<<<<< HEAD
---- Dizinler

INPUT_DIR = '/kaggle/input/strokedata1k/stroke_data'
DATASET_DIR = '/kaggle/working/dataset'
OUTPUT_DIR = '/kaggle/working/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

=======

# ---------------------- Model Oluşturma ----------------------
def build_model(base_model):
    with strategy.scope():
        inputs = Input(shape=(224,224,3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        outputs = Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs)

with strategy.scope():
    models = {
        'resnet':      build_model(ResNet101(weights='imagenet', include_top=False)),
        'vgg':         build_model(VGG16(weights='imagenet', include_top=False)),
        'densenet':    build_model(DenseNet121(weights='imagenet', include_top=False)),
        'efficientnet':build_model(EfficientNetB0(weights='imagenet', include_top=False))
    }

print("Models initialized.")
>>>>>>> 2373ce6bb31ba245e718595a3505d33f7dd6ac00
