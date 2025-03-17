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

#---- Dizinler----

INPUT_DIR = '/kaggle/input/strokedata1k/stroke_data'
DATASET_DIR = '/kaggle/working/dataset'
OUTPUT_DIR = '/kaggle/working/results'
os.makedirs(OUTPUT_DIR, exist_ok=True

# ---------------------- Veri Hazırlama ----------------------
def prepare_dataset():
    for split in ['train', 'value', 'test']:
        for cls in ['absent', 'present']:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)
            src_img = os.path.join(INPUT_DIR, split, cls)
            dst_img = os.path.join(DATASET_DIR, split, cls)
            if os.path.exists(src_img):
                for f in os.listdir(src_img):
                    shutil.copy(os.path.join(src_img, f), dst_img)

prepare_dataset()
print("Dataset preparation completed.")
# ---------------------- Data Generator ----------------------
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMG_SIZE = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

def create_generator(directory, datagen, shuffle=True):
    return datagen.flow_from_directory(
        directory, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=shuffle)

train_gen = create_generator(os.path.join(DATASET_DIR, 'train'), train_datagen, shuffle=True)
val_gen = create_generator(os.path.join(DATASET_DIR, 'value'), val_test_datagen, shuffle=False)
test_gen = create_generator(os.path.join(DATASET_DIR, 'test'), val_test_datagen, shuffle=False)

print("Data generators are ready.")


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

