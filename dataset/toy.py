import random

import numpy as np
import torch
from PIL import Image, ImageDraw
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import ssd_path
from dataset.utils import save_create_dir
import cv2


def create_line_image(path, size=(32, 32), straight=True):
    # create image with a random line (if straight=True, line is vertical)
    im = Image.new('RGB', size, 'green')
    if straight:
        x1 = random.randint(0, size[0])
        y1 = random.randint(0, size[1])
        x2 = x1
        y2 = random.randint(0, size[1])
    else:
        x1 = random.randint(0, size[0])
        y1 = random.randint(0, size[1])
        x2 = random.randint(0, size[0])
        y2 = random.randint(0, size[1])

    # draw line into image
    draw = ImageDraw.Draw(im)
    draw.line((x1, y1, x2, y2), fill=128)

    # plot image
    #plt.imshow(np.array(im))
    #plt.show()

    # save image
    im.save(path)

def create_color_image(path, size=(32,32), color='green'):
    im = Image.new('RGB', size, color)
    im.save(path)

def create_toy_dataset(path, size=5000, split=0.8):
    """
    creates a toy dataset at given path.
    Dataset has 2 classes.
    """
    # select kind of dataset
    #mode = 'color'
    mode = 'line'

    # randomly generating images
    assert split > 0 and split < 1
    img_shape = (32, 32)
    n_train = int(size * split)
    n_test = size - n_train

    # storing images
    path_train_straight = os.path.join(path, "train/straight/")
    path_train_tilt = os.path.join(path, "train/tilt/")
    path_test_straight = os.path.join(path, "val/straight/")
    path_test_tilt = os.path.join(path, "val/tilt/")
    save_create_dir(path_train_straight)
    save_create_dir(path_train_tilt)
    save_create_dir(path_test_straight)
    save_create_dir(path_test_tilt)

    # train images for class straight
    print("creating train images class 1/2")
    for i in tqdm(range(int(n_train / 2))):
        filename = os.path.join(path_train_straight, "image" + str(i) + ".PNG")
        if mode == 'line':
            create_line_image(filename, size=img_shape)
        elif mode == 'color':
            create_color_image(filename, size=img_shape, color='black')
    # train images for class tilt
    print("creating train images class 2/2")
    for i in tqdm(range(int(n_train / 2))):
        filename = os.path.join(path_train_tilt, "image" + str(i) + ".PNG")
        if mode == 'line':
            create_line_image(filename, size=img_shape, straight=False)
        elif mode == 'color':
            create_color_image(filename, size=img_shape, color='white')
    # test images for class straight
    print("creating test images class 1/2")
    for i in tqdm(range(int(n_test / 2))):
        filename = os.path.join(path_test_straight, "image" + str(i) + ".PNG")
        if mode == 'line':
            create_line_image(filename, size=img_shape)
        elif mode == 'color':
            create_color_image(filename, size=img_shape, color='black')
    # test images for class tilt
    print("creating test images class 2/2")
    for i in tqdm(range(int(n_test / 2))):
        filename = os.path.join(path_test_tilt, "image" + str(i) + ".PNG")
        if mode == 'line':
            create_line_image(filename, size=img_shape, straight=False)
        elif mode == 'color':
            create_color_image(filename, size=img_shape, color='white')

if __name__ == "__main__":
    create_toy_dataset(os.path.join(ssd_path, "toy"), size=5000)

