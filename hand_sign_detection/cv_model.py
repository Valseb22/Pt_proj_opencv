from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

ds_train_ = image_dataset_from_directory(
    'data/test',
    labels='inferred',
    label_mode='binary',
    image_size=[500, 500],
    interpolation='nearest',
    batch_size=3,
    shuffle=True,
)

ds_valid_ = image_dataset_from_directory(
    'data/test',
    labels='inferred',
    label_mode='binary',
    image_size=[500, 500],
    interpolation='nearest',
    batch_size=5,
    shuffle=False,
)

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

model = keras.Sequential([
    #pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation = 'sigmoid'),
    
])

optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=10,
)


history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();