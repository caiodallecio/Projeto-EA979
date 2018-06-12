import os
from random import shuffle
import matplotlib.pyplot as plt

import keras as k
import numpy as np
from keras import backend, layers, models
from keras.callbacks import TensorBoard
from PIL import Image, ImageDraw
from tqdm import tqdm


def import_train_images(limit: int = 0)->dict:
    print('import_train_images')
    files = os.listdir('../images')
    shuffle(files)
    limit = len(files) if not limit else limit
    files = files[:limit]
    ret = {}
    for filename in tqdm(files):
        string = '../images/'+filename
        tag = filename.split('.')[0]
        ret[tag] = (np.asarray(Image.open(string)).astype(
            'float32')/255).reshape(3, 256, 256)
    return ret


def import_train_labels()->dict:
    print('import_train_labels')
    files = os.listdir('../downloads')
    ret = {}
    for filename in tqdm(files):
        string = '../downloads/'+filename
        tag = filename.split('.')[0]
        ret[tag] = (np.asarray(Image.open(string)).astype(
            'float32')/255).reshape(3, 256, 256)
    return ret


def create_memory_dataset(images: dict, labels: dict):
    print('create_memory_dataset')
    dataset = []
    for label, image in tqdm(images.items()):
        dataset.append((image, labels[label.split('_')[0]]))
    return dataset


def split_dataset(dataset, ratio=0.8):
    shuffle(dataset)
    size = len(dataset)
    train, test = dataset[:int(ratio*size)], dataset[int(ratio*size):]
    return train, test


def create_model(input_shape):
    network = models.Sequential()
    network.add(layers.Conv2D(3, (3, 3), activation='relu',
                              padding='same', input_shape=input_shape))
    network.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    network.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    network.add(layers.Dense(64))
    network.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    network.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    network.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))

    network.add(layers.Conv2D(
        3, (3, 3), activation='sigmoid', padding='same'))

    network.compile(optimizer='adam', loss='binary_crossentropy')

    network.summary()
    return network


if __name__ == '__main__':
    images = import_train_images(limit=1200)
    labels = import_train_labels()
    dataset = create_memory_dataset(images, labels)
    train, test = split_dataset(dataset)
    autoencoder = create_model((256, 256, 3))

    x_train_noisy, x_train = [x[0] for x in train], [x[1] for x in train]
    x_test_noisy, x_test = [x[0] for x in test], [x[1] for x in test]

    x_train_noisy = np.reshape(
        x_train_noisy, (len(x_train_noisy), 256, 256, 3))
    x_train = np.reshape(x_train, (len(x_train), 256, 256, 3))

    x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), 256, 256, 3))
    x_test = np.reshape(x_test, (len(x_test), 256, 256, 3))

    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(x_train_noisy[i].reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(x_train[i].reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=50,
                    batch_size=40,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    decoded_imgs = autoencoder.predict(x_test_noisy)

    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(x_test_noisy[i].reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(decoded_imgs[i].reshape(256, 256, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    autoencoder.save('trained_network_v3_multi_size.h5')
