import argparse
import json
import math
import os
import random
import numpy as np
from os import listdir, remove, rename

from google_images_download import google_images_download
from keras.utils import Sequence
from PIL import Image, ImageDraw
from tqdm import tqdm

DOWNLOAD_DIR = "../downloads"
OUT_FILE = "../dataset.json"
OUT_SIZE = (128, 128)

# Downloads images


class Images(Sequence):

    def __init__(self, dataset, batchsize):
        random.shuffle(dataset)
        self.dataset = dataset
        self.batch_size = batchsize

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = [Image.open(DOWNLOAD_DIR+'/'+file_name[0]).resize((128, 128))
                   for file_name in batch]
        batch_x = []
        for image, data in zip(batch_y, batch):
            image = image.copy()
            draw = ImageDraw.Draw(image)
            lines = data[1]
            for line in lines:
                draw.line((line[1], line[2], line[3], line[4]),
                          (255, 255, 255), line[0])
            batch_x.append(np.array(image).astype('float32')/255)
        batch_y = [np.array(y).astype('float32')/255 for y in batch_y]
        return np.array(batch_x), np.array(batch_y)


def download_images(download_dir: str, out_size: tuple, times: int):
    """Download images"""
    response = google_images_download.googleimagesdownload()
    config = {"keywords": "family photo",
              "limit": times,
              "aspect_ratio": "square",
              "type": "photo",
              "output_directory": download_dir,
              "no_directory": True,
              "format": "jpg",
              "chromedriver": "/usr/bin/chromedriver"}
    response.download(config)

    for i, filename in enumerate(listdir(download_dir)):
        name = download_dir+"/"+"%04d.jpg" % i
        filepath = download_dir+"/"+filename
        rename(filepath, name)

    for filename in listdir(download_dir):
        try:
            filepath = download_dir+"/"+filename
            image = Image.open(filepath)
            image = image.resize(out_size, Image.LANCZOS)
            image.save(filepath)
        except OSError:
            filepath = download_dir+"/"+filename
            remove(filepath)


def modify_images(download_dir: str, out_file: str, out_size: tuple, times: int):
    ret = []
    n_shapes = (1, 3)
    line_size = (4, 16)
    width = 1
    for filename in tqdm(listdir(download_dir)):
        versions_list = []
        for _ in range(0, times):
            shape_list = []
            num_shapes = random.randint(*n_shapes)
            for _ in range(0, num_shapes):
                size = random.randint(*line_size)
                # line_angle = random.uniform(0, 2*math.pi)
                line_x1 = random.randint(0, out_size[0])
                line_y1 = random.randint(0, out_size[0])
                line_x2 = random.randint(0, out_size[0])
                line_y2 = random.randint(0, out_size[0])
                shape_list.append((width, line_x1, line_x2, line_y1, line_y2))
            versions_list.append(shape_list)
        ret.extend([(filename, version) for version in versions_list])
    json.dump(ret, open(out_file, 'w'))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Dowload images from google and create dataset')
    PARSER.add_argument('--download', type=int)
    PARSER.add_argument('--dataset', type=int)
    ARGS = PARSER.parse_args()
    print(ARGS)
    if ARGS.download:
        download_images(DOWNLOAD_DIR, OUT_SIZE, ARGS.download)
    if ARGS.dataset:
        modify_images(DOWNLOAD_DIR, OUT_FILE, OUT_SIZE, ARGS.dataset,)
