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

