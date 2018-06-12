import math
import random
import os
from os import listdir, remove, rename
import argparse
from tqdm import tqdm

from google_images_download import google_images_download
from PIL import Image, ImageDraw

DOWNLOAD_DIR = "../downloads"
OUT_DIR = "../images"
OUT_SIZE = (256, 256)

# Downloads images


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


def modify_images(download_dir: str, out_dir: str, out_size: tuple, times: int):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    n_shapes = (2, 6)
    line_size = (15, 150)
    line_width = (4, 10)
    ellipse_height = (15, 30)
    ellipse_width = (15, 30)
    for filename in tqdm(listdir(download_dir)):
        filepath = download_dir+"/"+filename
        for i in range(0, times):
            image = Image.open(filepath)
            draw = ImageDraw.Draw(image)
            num_shapes = random.randint(*n_shapes)
            for _ in range(0, num_shapes):
                shape = random.randint(0, 1)
                # Lines
                if shape == 0:
                    size = random.randint(*line_size)
                    line_angle = random.uniform(0, 2*math.pi)
                    width = random.randint(*line_width)
                    line_x1 = random.randint(0, out_size[0])
                    line_y1 = random.randint(0, out_size[0])
                    line_x2 = line_x1 + size*math.cos(line_angle)
                    line_y2 = line_y1 + size*math.sin(line_angle)
                    draw.line((line_x1, line_y1, line_x2, line_y2),
                              (255, 255, 255),
                              width
                              )
                # ellipses
                if shape == 1:
                    width = random.randint(*ellipse_width)
                    height = random.randint(*ellipse_height)
                    ellipse_x1 = random.randint(0, out_size[0])
                    ellipse_y1 = random.randint(0, out_size[0])
                    ellipse_x2 = ellipse_x1 + width
                    ellipse_y2 = ellipse_y1 + height
                    draw.ellipse((ellipse_x1,
                                  ellipse_y1,
                                  ellipse_x2,
                                  ellipse_y2),
                                 (255, 255, 255))

            image.save(out_dir+"/"+str(filename).split('.')
                       [0]+"_"+str(i)+'.jpg', "JPEG")


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
        modify_images(DOWNLOAD_DIR, OUT_DIR, OUT_SIZE, ARGS.dataset,)
