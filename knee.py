# %%
import numpy as np
import pandas as pd
import time
import shutil
import pathlib
import itertools
from PIL import Image
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings("ignore")
from preprocessing import train_gen_new, valid_gen_new, test_gen_new
print ('check')
# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %%
import keras
from tensorflow.keras.applications import EfficientNetV2L, DenseNet121
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5,
restore_best_weights=True)
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, GaussianNoise, MultiHeadAttention, Reshape)
from tensorflow.keras.optimizers import Adam

input_shape = (224, 224, 3)
num_classes=3
learning_rate=1e-4
inputs = Input(shape=input_shape, name="Input_Layer")

base_model = EfficientNetV2L(weights='imagenet', input_tensor=inputs, include_top=False)
base_model.trainable = False
x = base_model.output
height, width, channels = x.shape[1], x.shape[2], x.shape[3]
x = Reshape((height * width, channels), name="Reshape_to_Sequence")(x)
x = MultiHeadAttention(num_heads=8, key_dim=channels, name="Multi_Head_Attention")(x, x)
x = Reshape((height, width, channels), name="Reshape_to_Spatial")(x)
x = GaussianNoise(0.25, name="Gaussian_Noise")(x)
x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
x = Dense(512, activation="relu", name="FC_512")(x)
x = BatchNormalization(name="Batch_Normalization")(x)
x = Dropout(0.25, name="Dropout")(x)
outputs = Dense(num_classes, activation="softmax",name="Output_Layer")(x)
model = Model(inputs=inputs, outputs=outputs, name="Xception_with_Attention")
model.compile(
optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

cnn_model = model

# %%
history = cnn_model.fit( train_gen_new,
                         validation_data=valid_gen_new,
                         epochs=250,
                         callbacks=[early_stopping],
                         verbose=1)

y_pred = cnn_model.predict(valid_gen_new)
y_true = valid_gen_new.labels

def ppo_loss(y_true, y_pred):
    epsilon = 0.2
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
    selected_probs = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
    old_selected_probs = tf.reduce_sum(tf.stop_gradient(y_pred) * y_true_one_hot, axis=-1)
    ratio = selected_probs / (old_selected_probs + 1e-10)
    clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
    loss = -tf.reduce_mean(tf.minimum(ratio, clipped_ratio))
    return loss

ppo_loss_value = ppo_loss(y_true, y_pred)
print("\nPPO Loss on Validation Data:", ppo_loss_value.numpy())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show(block=False)
plt.pause(5)
plt.close()

test_labels = test_gen_new.classes
predictions = cnn_model.predict(test_gen_new)
predicted_classes = np.argmax(predictions, axis=1)

report = classification_report(test_labels, predicted_classes,
target_names=list(test_gen_new.class_indices.keys()))
print(report)

conf_matrix = confusion_matrix(test_labels, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=list(test_gen_new.class_indices.keys()),
yticklabels=list(test_gen_new.class_indices.keys()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show(block=False)
plt.pause(5)
plt.close()