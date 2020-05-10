from tensorflow.keras.applications import vgg19
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from constants import *


def load(path):
    img = cv.imread(path)
    img = cv.resize(img, (ncols, nrows))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img


def preprocess(img):
    img = vgg19.preprocess_input(img)
    return img


def deprocess(img):
    img = img[0]
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def load_images(base, style):
    base_img = load(base)
    style_img = load(style)

    base_input = preprocess(base_img)
    style_input = preprocess(style_img)

    return base_input, style_input
