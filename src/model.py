import os
from random import shuffle
import matplotlib.pyplot as plt

import keras as k
import numpy as np
import json
from keras import backend, layers, models
from keras.callbacks import TensorBoard
from keras.models import load_model
from PIL import Image, ImageDraw
from tqdm import tqdm
from genset import Images


def create_model(input_shape):
    network = models.Sequential()

    network.add(layers.Conv2D(8, (3, 3), activation='tanh',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(16, (3, 3), activation='tanh',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(32, (3, 3), activation='tanh',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(64, (3, 3), activation='tanh',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(32, (3, 3), activation='tanh',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(16, (3, 3), activation='tanh',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(8, (3, 3), activation='tanh',
                              padding='same', input_shape=input_shape))
   
    network.add(layers.Conv2D(
        3, (3, 3), activation='sigmoid', padding='same'))

    network.compile(optimizer='adam', loss='mse')

    network.summary()
    return network


if __name__ == '__main__':
    if os.path.isfile('autoencoder2.h5'):
        autoencoder = load_model('autoencoder2.h5')
    else:
        autoencoder = create_model((256, 256, 3))

    autoencoder.fit_generator(
        Images(json.load(open('../dataset.json')), 40, '../downloads'),
        validation_data=Images(json.load(open('../test.json')), 10, '../test'),
        epochs=200,
        use_multiprocessing=True,
        workers=16)

    imgs = Images(json.load(open('../test.json')), 1, '../test')
    for i in range(10):
        a, _ = imgs[i]
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

    autoencoder.save('autoencoder2.h5')
