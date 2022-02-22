import itertools
import os
import sys
import warnings
from glob import glob
from math import ceil, sqrt
from os import listdir
from os.path import isfile, join

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tensorflow as tf
import tifffile as tif
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import (
    Activation,
    Add,
    Average,
    AveragePooling2D,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    Multiply,
    SeparableConv2D,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
    dot,
    multiply,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean, rescale, resize
from skimage.util.shape import view_as_windows
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

from loss import *
from model import *
from model import msrf
from utils import *

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

np.random.seed(42)
create_dir("files")

train_path = "data/kdsb/train/"
valid_path = "data/kdsb/valid/"

## Training
train_x = sorted(glob(os.path.join(train_path, "images", "*.jpg")))
train_y = sorted(glob(os.path.join(train_path, "masks", "*.jpg")))

## Shuffling
train_x, train_y = shuffling(train_x, train_y)
train_x = train_x
train_y = train_y

## Validation
valid_x = sorted(glob(os.path.join(valid_path, "images", "*.jpg")))
valid_y = sorted(glob(os.path.join(valid_path, "masks", "*.jpg")))


print("final training set length", len(train_x), len(train_y))
print("final valid set length", len(valid_x), len(valid_y))

import random

X_tot_val = [get_image(sample_file, 256, 256) for sample_file in valid_x]
X_val, edge_x_val = [], []
print(len(X_tot_val))
for i in range(0, len(valid_x)):
    X_val.append(X_tot_val[i][0])
    edge_x_val.append(X_tot_val[i][1])
X_val = np.array(X_val).astype(np.float32)
edge_x_val = np.array(edge_x_val).astype(np.float32)
edge_x_val = np.expand_dims(edge_x_val, axis=3)
Y_tot_val = [get_image(sample_file, 256, 256, gray=True) for sample_file in valid_y]
Y_val, edge_y = [], []
for i in range(0, len(valid_y)):
    Y_val.append(Y_tot_val[i][0])
Y_val = np.array(Y_val).astype(np.float32)

Y_val = np.expand_dims(Y_val, axis=3)


def train(epochs, batch_size, output_dir, model_save_dir):

    batch_count = int(len(train_x) / batch_size)
    max_val_dice = -1
    G = msrf()
    G.summary()
    optimizer = get_optimizer()
    G.compile(
        optimizer=optimizer,
        loss={"x": seg_loss, "edge_out": "binary_crossentropy", "pred4": seg_loss, "pred2": seg_loss},
        loss_weights={"x": 2.0, "edge_out": 1.0, "pred4": 1.0, "pred2": 1.0},
    )
    for e in range(1, epochs + 1):
        print("-" * 15, "Epoch %d" % e, "-" * 15, batch_size)
        # sp startpoint
        for sp in range(0, batch_count, 1):
            if (sp + 1) * batch_size > len(train_x):
                batch_end = len(train_x)
            else:
                batch_end = (sp + 1) * batch_size
            X_batch_list = train_x[(sp * batch_size) : batch_end]
            Y_batch_list = train_y[(sp * batch_size) : batch_end]
            X_tot = [get_image(sample_file, 256, 256) for sample_file in X_batch_list]
            X_batch, edge_x = [], []
            for i in range(0, batch_size):
                X_batch.append(X_tot[i][0])
                edge_x.append(X_tot[i][1])
            X_batch = np.array(X_batch).astype(np.float32)
            edge_x = np.array(edge_x).astype(np.float32)
            Y_tot = [get_image(sample_file, 256, 256, gray=True) for sample_file in Y_batch_list]
            Y_batch, edge_y = [], []
            for i in range(0, batch_size):
                Y_batch.append(Y_tot[i][0])
                edge_y.append(Y_tot[i][1])
            Y_batch = np.array(Y_batch).astype(np.float32)
            edge_y = np.array(edge_y).astype(np.float32)
            Y_batch = np.expand_dims(Y_batch, axis=3)
            edge_y = np.expand_dims(edge_y, axis=3)
            edge_x = np.expand_dims(edge_x, axis=3)
            G.train_on_batch([X_batch, edge_x], [Y_batch, edge_y, Y_batch, Y_batch])

        y_pred, _, _, _ = G.predict([X_val, edge_x_val], batch_size=5)
        y_pred = (y_pred >= 0.5).astype(int)
        res = mean_dice_coef(Y_val, y_pred)
        if res > max_val_dice:
            max_val_dice = res
            G.save("kdsb_ws.h5")
            print("New Val_Dice HighScore", res)


model_save_dir = "./model/"
output_dir = "./output/"
train(125, 4, output_dir, model_save_dir)
