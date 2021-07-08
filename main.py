"""
A NN exercise to recognise mathematical symbols(+,-,=) inspired by MNIST
Dataset created with https://github.com/AxelThevenot/Python-Interface-to-Create-Handwritten-dataset
"""
import os
import tensorflow as tf
from keras.layers import Reshape
from tensorflow import keras
from tensorflow.python.keras.layers import Rescaling

SEED = 1984
DATA_PATH = 'data/mathsymbols'
MODEL_PATH = 'models/mathsymbols'


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


if os.path.isdir(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    train_ds = get_dataset()
    val_ds = get_dataset(subset='validation')

    model = keras.Sequential([
        Reshape((28 * 28,)),
        Rescaling(1 / 255),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200
    )
    model.save(MODEL_PATH)

new_test_image = tf.keras.preprocessing.image.load_img(
    'testdata/mathsymbols/-.png', color_mode="grayscale", target_size=(28, 28))
new_test_image = keras.preprocessing.image.img_to_array(new_test_image)
new_test_image = tf.expand_dims(new_test_image, 0)  # Create a batch

predictions = model.predict(new_test_image)
print(predictions)
score = tf.nn.softmax(predictions[0])
print(score)
