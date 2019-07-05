import os
import numpy as np

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def crop(img):
    return img[34:194,:]

def preprocess(img):
    return downsample(crop(to_grayscale(img))) / 255.0

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
