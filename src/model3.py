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


if __name__ == '__main__':
    autoencoder = load_model('autoencoder3.h5')
    
    imgs = Images(json.load(open('../dataset.json')), 1, '../downloads')
    for i in range(10):
        X, Y = imgs[i]
        # display original
        ax = plt.subplot(3, 10, i + 1)
        plt.imshow(X.reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        print(1,autoencoder.evaluate(X,Y))


        # display reconstruction
        b = autoencoder.predict(X)
        ax = plt.subplot(3, 10, i + 1 + 10)
        plt.imshow(b.reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        print(2,autoencoder.evaluate(b,Y))

        # display reconstruction
        c = autoencoder.predict(b)
        ax = plt.subplot(3, 10, i + 1 + 20)
        plt.imshow(c.reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        print(3,autoencoder.evaluate(c,Y))

    plt.show()
