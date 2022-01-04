import numpy as np
import os, warnings
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.ops.variables import validate_synchronization_aggregation_trainable
from tensorflow.python.platform.tf_logging import warning
import tensorflow_hub as hub
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow import keras
from tensorflow.keras import layers

# Seed setup
def set_seed(seed = 31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    os.environ['TF_DETERMINISTIC_OPS']='1'

set_seed()

# Set pyplot configs
plt.rc('figure',autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap = 'magma')
warnings.filterwarnings('ignore')

# Load data - train and valid datasets

# path with double \\
train_path = 'C:\\Users\\User\\VS Code\\Computer_Vision_Projects\\CNN_CarTruck_Classifier\\train'
train_ds = image_dataset_from_directory(
    directory=train_path,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)

# Raw path with r in front
valid_path = r'C:\Users\User\VS Code\Computer_Vision_Projects\CNN_CarTruck_Classifier\valid'
valid_ds = image_dataset_from_directory(
    directory=valid_path,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Convert to float
def convert_float(image, label):
    image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = (
    train_ds
    .map(convert_float)
    .cache()
    .prefetch(buffer_size = AUTOTUNE)
)

valid_ds = (
    valid_ds
    .map(convert_float)
    .cache()
    .prefetch(buffer_size = AUTOTUNE)
)

print("Datasets are ready. . .")