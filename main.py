"""
A NN exercise to recognise mathematical symbols(+,-,=) inspired by MNIST
Dataset created with https://github.com/AxelThevenot/Python-Interface-to-Create-Handwritten-dataset
"""
import os

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Rescaling, Reshape
from tensorflow import keras

SEED = 1984
DATA_PATH = 'data/mathsymbols'
MODEL_PATH = 'models/mathsymbols'

# TODO  Ideally this should not be hardcoded! Save and load from the ds! Can be obtained from traind_ds.class_names. The saved model doesn't include it.
CLASS_NAMES = ['+', '-', '=']

def get_dataset(subset='training'):
    return tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH,
        label_mode='categorical',
        image_size=(28, 28),
        color_mode="grayscale",
        shuffle=True,
        seed=SEED,
        validation_split=0.20,
        subset=subset)

def build_model():
    model = keras.Sequential([
        Reshape((28 * 28,)),
        Rescaling(1 / 255),
        Dense(200, activation='relu'),
        Dense(20, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    train_ds = get_dataset()
    val_ds = get_dataset(subset='validation')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100
    )
    return model

if os.path.isdir(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    model = build_model()
    model.save(MODEL_PATH)

new_test_image = keras.preprocessing.image.load_img(
    'testdata/mathsymbols/+.png', color_mode="grayscale", target_size=(28, 28))
new_test_image = keras.preprocessing.image.img_to_array(new_test_image)
new_test_image = tf.expand_dims(new_test_image, 0)  # Create a batch

predictions = model.predict(new_test_image)
print(f"Predictions: {predictions}")
score = tf.nn.softmax(predictions[0])

print(f"Score: {score}")
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
)