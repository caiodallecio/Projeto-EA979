import json
import os
from random import shuffle

import keras as k
import matplotlib.pyplot as plt
import numpy as np
from keras import backend, layers, models
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from PIL import Image, ImageDraw
from tqdm import tqdm

from genset import Images


def create_model(input_shape):
    network = models.Sequential()

    network.add(layers.Conv2D(4, (3, 3), activation='relu',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(8, (3, 3), activation='relu',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(16, (3, 3), activation='relu',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(32, (3, 3), activation='relu',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(64, (3, 3), activation='relu',
                              padding='same', input_shape=input_shape))

    network.add(layers.Deconv2D(64, (3, 3), activation='relu',
                                padding='same', input_shape=input_shape))
    network.add(layers.Deconv2D(32, (3, 3), activation='relu',
                                padding='same', input_shape=input_shape))
    network.add(layers.Deconv2D(16, (3, 3), activation='relu',
                                padding='same', input_shape=input_shape))
    network.add(layers.Deconv2D(8, (3, 3), activation='relu',
                                padding='same', input_shape=input_shape))
    network.add(layers.Deconv2D(4, (3, 3), activation='relu',
                                padding='same', input_shape=input_shape))
    network.add(layers.Deconv2D(
        3, (3, 3), activation='sigmoid', padding='same'))

    network.compile(optimizer='adam', loss='mse')

    network.summary()
    return network


if __name__ == '__main__':
    if os.path.isfile('autoencoder3.h5'):
        autoencoder = load_model('autoencoder3.h5')
    else:
        autoencoder = create_model((256, 256, 3))

    train = Images(json.load(open('../dataset.json')), 10, '../downloads')
    validation = Images(json.load(open('../test.json')), 10, '../test')
    test = Images(json.load(open('../test.json')), 1, '../test')

    autoencoder.fit_generator(
        train,
        validation_data=validation,
        callbacks=[
            ModelCheckpoint('checkpoint.hdf5', save_best_only=True),
            ReduceLROnPlateau(patience=5)
        ],
        epochs=30,
        use_multiprocessing=True,
        workers=os.cpu_count())

    for i in range(10):
        a, _ = test[i]
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(a.reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        b = autoencoder.predict(a)
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(b.reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    autoencoder.save('autoencoder3.h5')
