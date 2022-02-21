import csv
import json
import os
import re
from glob import glob
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(123)
import itertools
import os
import warnings
from glob import glob
from math import ceil, sqrt

import cv2
import keras
import numpy as np
import skimage.io
import tensorflow as tf
import tifffile as tif
from keras import backend as K
from keras import regularizers
from keras.applications import (
    VGG16,
    VGG19,
    DenseNet121,
    DenseNet169,
    InceptionResNetV2,
    ResNet50,
)
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
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
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import (
    to_categorical,  # used for converting labels to one-hot-encoding
)
from PIL import Image
from skimage.color import rgb2gray
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    black_tophat,
    closing,
    convex_hull_image,
    dilation,
    erosion,
    opening,
    skeletonize,
    square,
    white_tophat,
)
from skimage.transform import downscale_local_mean, rescale, resize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import *
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()


def create_dir(path):
    """Create a directory."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def read_data(x, y):
    """Read the image and mask from the given path."""
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    return image, mask


def read_params():
    """Reading the parameters from the JSON file."""
    with open("params.json", "r") as f:
        data = f.read()
        params = json.loads(data)
        return params


def load_data(path):
    """Loading the data from the given path."""
    images_path = os.path.join(path, "image/*")
    masks_path = os.path.join(path, "mask/*")

    images = glob(images_path)
    masks = glob(masks_path)

    return images, masks


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = "same"
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = "valid"

    if not depth_activation:
        x = Activation("relu")(x)
    x = DepthwiseConv2D(
        (kernel_size, kernel_size),
        strides=(stride, stride),
        dilation_rate=(rate, rate),
        padding=depth_padding,
        use_bias=False,
        name=prefix + "_depthwise",
    )(x)
    x = BatchNormalization(name=prefix + "_depthwise_BN", epsilon=epsilon)(x)
    if depth_activation:
        x = Activation("relu")(x)
    x = Conv2D(filters, (1, 1), padding="same", use_bias=False, name=prefix + "_pointwise")(x)
    x = BatchNormalization(name=prefix + "_pointwise_BN", epsilon=epsilon)(x)
    if depth_activation:
        x = Activation("relu")(x)

    return x


def get_image(image_path, image_size_wight, image_size_height, gray=False):
    # load image
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if gray == True:
        img = img.convert("L")
    # center crop
    img_center_crop = img
    # resize
    img_resized = img
    edge = cv2.Canny(np.asarray(np.uint8(img_resized)), 10, 1000)

    flag = False
    # convert to numpy and normalize
    img_array = np.asarray(img_resized).astype(np.float32) / 255.0
    edge = np.asarray(edge).astype(np.float32) / 255.0
    # print(img_array)
    if gray == True:
        img_array = (img_array >= 0.5).astype(int)
    img.close()
    return img_array, edge


from glob import glob

np.random.seed(42)
create_dir("files")

train_path = "../data/isic/train/"
valid_path = "../data/isic/valid/"

## Training
train_x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
train_y = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))

## Shuffling
train_x, train_y = shuffling(train_x, train_y)
train_x = train_x
train_y = train_y

## Validation
valid_x = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
valid_y = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))


print("final training set length", len(train_x), len(train_y))
import random

X_tot_val = [get_image(sample_file, 288, 384) for sample_file in valid_x]
X_val, edge_x_val = [], []
print(len(X_tot_val))
for i in range(0, len(valid_x)):
    X_val.append(X_tot_val[i][0])
    edge_x_val.append(X_tot_val[i][1])
X_val = np.array(X_val).astype(np.float32)
edge_x_val = np.array(edge_x_val).astype(np.float32)
edge_x_val = np.expand_dims(edge_x_val, axis=3)
Y_tot_val = [get_image(sample_file, 288, 384, gray=True) for sample_file in valid_y]
Y_val, edge_y = [], []
for i in range(0, len(valid_y)):
    Y_val.append(Y_tot_val[i][0])
Y_val = np.array(Y_val).astype(np.float32)

Y_val = np.expand_dims(Y_val, axis=3)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, "float32")
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), "float32")
    intersection = y_true_f * y_pred_f
    score = 2.0 * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - score


def dice_coefficient_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1.0 - dice_loss(y_true, y_pred))


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    logit_y_pred = K.log(y_pred / (1.0 - y_pred))
    loss = weight * (
        logit_y_pred * (1.0 - y_true) + K.log(1.0 + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.0)
    )
    return K.sum(loss) / K.sum(weight)


def weighted_dice(y_true, y_pred):
    smooth = 1.0
    w, m1, m2 = 0.7, y_true, y_pred
    intersection = m1 * m2
    score = (2.0 * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return K.sum(score)


def weighted_dice_loss(y_true, y_pred):
    smooth = 1.0
    w, m1, m2 = 0.7, y_true, y_pred
    intersection = m1 * m2
    score = (2.0 * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1.0 - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(y_true, pool_size=(50, 50), strides=(1, 1), padding="same", pool_mode="avg")
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5.0 * K.exp(-5.0 * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= w0 / w1
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true[:, :, :, 0])
    y_pred_f = K.flatten(y_pred[:, :, :, 0])
    intersection = K.sum(y_true_f * y_pred_f)
    d1 = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return d1


def spatial_att_block(x, intermediate_channels):
    out = Conv2D(intermediate_channels, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same")(out)
    out = Activation("sigmoid")(out)
    return out


def resblock(x, ip_channels, op_channels, stride=(1, 1)):
    residual = x
    out = Conv2D(op_channels, kernel_size=(3, 3), strides=stride, padding="same")(x)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(op_channels, kernel_size=(3, 3), strides=stride, padding="same")(x)
    out = BatchNormalization()(out)
    out = Add()([out, residual])
    out = Activation("relu")(out)
    return out


def dual_att_blocks(skip, prev, out_channels):
    up = Conv2DTranspose(out_channels, 4, strides=(2, 2), padding="same")(prev)
    up = BatchNormalization()(up)
    up = Activation("relu")(up)
    inp_layer = Concatenate()([skip, up])
    inp_layer = Conv2D(out_channels, 3, strides=(1, 1), padding="same")(inp_layer)
    inp_layer = BatchNormalization()(inp_layer)
    inp_layer = Activation("relu")(inp_layer)
    se_out = se_block(inp_layer, out_channels)
    sab = spatial_att_block(inp_layer, out_channels // 4)
    # sab = Add()([sab,1])
    sab = Lambda(lambda y: y + 1)(sab)
    final = Multiply()([sab, se_out])
    return final


def gsc(input_features, gating_features, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1):
    x = Concatenate()([input_features, gating_features])
    x = BatchNormalization()(x)
    x = Conv2D(in_channels + 1, (1, 1), strides=(1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(1, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    # x = sigmoid(x)
    return x


def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch // ratio, activation="relu")(x)
    x = Dense(ch, activation="sigmoid")(x)
    return Multiply()([in_block, x])


def Attention_B(X, G, k):
    FL = int(X.shape[-1])
    init = RandomNormal(stddev=0.02)
    theta = Conv2D(k, (2, 2), strides=(2, 2), padding="same")(X)
    Phi = Conv2D(k, (1, 1), strides=(1, 1), padding="same", use_bias=True)(G)

    ADD = Add()([theta, Phi])

    # ADD = LeakyReLU(alpha=0.1)(ADD)
    ADD = Activation("relu")(ADD)

    # Psi = Conv2D(FL,(1,1), strides = (1,1), padding="same",kernel_initializer=init)(ADD)
    Psi = Conv2D(1, (1, 1), strides=(1, 1), padding="same", kernel_initializer=init)(ADD)
    Psi = Activation("sigmoid")(Psi)
    Up = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding="valid")(Psi)

    # Psi = Activation('tanh')(Psi)

    # Up = Conv2DTranspose(FL, (2,2), strides=(2, 2), padding='valid')(Psi)

    Final = Multiply()([X, Up])
    # Final = Conv2D(1, (1,1), strides = (1,1), padding="same",kernel_initializer=init)(Final)
    # Final = Conv2D(FL, (1,1), strides = (1,1), padding="same",kernel_initializer=init)(Final)
    Final = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-5)(Final)
    print(Final.shape)
    return Final


def Unet3(input_shape, n_filters, kernel=(3, 3), strides=(1, 1), pad="same"):
    x = input_shape
    conv1 = Conv2D(n_filters, kernel_size=kernel, strides=strides, padding=pad)(input_shape)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)

    conv2 = Conv2D(n_filters, kernel_size=kernel, strides=strides, padding=pad)(conv1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)

    x = Conv2D(n_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)

    return Add()([x, conv2])


def Up3(input1, input2, kernel=(3, 3), stride=(1, 1), pad="same"):
    # up = UpSampling2D(2)(input2)
    up = Conv2DTranspose(int(input1.shape[-1]), (1, 1), strides=(2, 2), padding="same")(input2)
    up = Concatenate()([up, input1])

    # up1 = BatchNormalization()(up)
    # up1 =  LeakyReLU(alpha=0.25)(up1)
    # up1 = Conv2D(int(input1.shape[-1]),kernel_size=(3,3),strides=(1,1),padding='same')(up1)
    # up1 = BatchNormalization()(up1)
    # up1 =  LeakyReLU(alpha=0.25)(up1)
    # up1 = Conv2D(int(input1.shape[-1]),kernel_size=(3,3),strides=(1,1),padding='same')(up1)
    # up2 = Add()([up1,up])
    return up

    return Unet3(up, int(input1.shape[-1]), kernel, stride, pad)


def gatingSig(input_shape, n_filters, kernel=(1, 1), strides=(1, 1), pad="same"):
    conv = Conv2D(n_filters, kernel_size=kernel, strides=strides, padding=pad)(input_shape)
    conv = BatchNormalization(axis=-1)(conv)
    return LeakyReLU(alpha=0.1)(conv)


def DSup(x, var):
    d = Conv2D(1, (1, 1), strides=(1, 1), padding="same")(x)
    d = UpSampling2D(var)(d)
    return d


def DSup1(x, var):
    d = Conv2D(1, (2, 2), strides=(2, 2), padding="same")(x)
    d = UpSampling2D(var)(d)
    return d


# Keras
ALPHA = 0.5
BETA = 0.5
GAMMA = 1


def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):

    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky


def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):

    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    FocalTversky = K.pow((1 - Tversky), gamma)

    return FocalTversky


def sau(input_size=(384, 512, 3), input_size_2=(384, 512, 1)):
    n_labels = 1
    feature_scale = 8
    # input_shape= Image.shape
    filters = [64, 128, 256, 512, 1024]
    atrous_rates = (6, 12, 18)
    n_labels = 1
    feature_scale = 8
    # input_shape= Image.shape
    filters = [64, 128, 256, 512, 1024]

    inputs_img = Input(input_size)
    canny = Input(input_size_2, name="checkdim")
    n11 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs_img)
    n11 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n11)
    n11 = BatchNormalization()(n11)
    n11 = se_block(n11, 32)

    n12 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n11)
    n12 = BatchNormalization()(n12)
    # here
    n12 = Add()([n12, n11])
    pred1 = Conv2D(1, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")(n12)

    pool1 = MaxPooling2D(pool_size=(2, 2))(n11)
    pool1 = Dropout(0.2)(pool1)
    n21 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    n21 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n21)
    n21 = BatchNormalization()(n21)
    n21 = se_block(n21, 64)
    # here

    n22 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n21)
    n22 = BatchNormalization()(n22)
    n22 = Add()([n22, n21])

    pool2 = MaxPooling2D(pool_size=(2, 2))(n21)
    pool2 = Dropout(0.2)(pool2)

    n31 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    n31 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n31)
    n31 = BatchNormalization()(n31)
    n31 = se_block(n31, 128)

    n32 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n31)
    n32 = BatchNormalization()(n32)
    n32 = Add()([n32, n31])

    pool3 = MaxPooling2D(pool_size=(2, 2))(n31)
    pool3 = Dropout(0.2)(pool3)
    n41 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    n41 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n41)
    n41 = BatchNormalization()(n41)
    #############################################ASPP
    shape_before = n41.shape

    #############################################

    n12, n22 = RDDB(n11, n21, 32, 64, 16)
    pred2 = Conv2D(1, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")(n12)

    n32, n42 = RDDB(n31, n41, 128, 256, 64)

    n12, n22 = RDDB(n12, n22, 32, 64, 16)
    pred3 = Conv2D(1, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")(n12)

    n32, n42 = RDDB(n32, n42, 128, 256, 64)

    n22, n32 = RDDB(n22, n32, 64, 128, 32)

    n13, n23 = RDDB(n12, n22, 32, 64, 16)

    n33, n43 = RDDB(n32, n42, 128, 256, 64)

    n23, n33 = RDDB(n23, n33, 64, 128, 32)

    n13, n23 = RDDB(n13, n23, 32, 64, 16)

    n33, n43 = RDDB(n33, n43, 128, 256, 64)

    n13 = Lambda(lambda x: x * 0.4)(n13)
    n23 = Lambda(lambda x: x * 0.4)(n23)
    n33 = Lambda(lambda x: x * 0.4)(n33)
    n43 = Lambda(lambda x: x * 0.4)(n43)

    n13, n23 = Add()([n11, n13]), Add()([n21, n23])
    n33, n43 = Add()([n31, n33]), Add()([n41, n43])

    ###############Shape Stream

    d0 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="same")(n23)
    ss = keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")(d0)
    ss = resblock(ss, 32, 32)
    c3 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same")(n33)
    c3 = keras.layers.UpSampling2D(size=(4, 4), data_format=None, interpolation="bilinear")(c3)
    ss = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding="same")(ss)
    ss = gsc(ss, c3, 32, 32)
    ss = resblock(ss, 16, 16)
    ss = Conv2D(8, kernel_size=(1, 1), strides=(1, 1), padding="same")(ss)
    c4 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same")(n43)
    c4 = keras.layers.UpSampling2D(size=(8, 8), data_format=None, interpolation="bilinear")(c4)
    ss = gsc(ss, c4, 16, 16)
    ss = resblock(ss, 8, 8)
    ss = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding="same")(ss)
    ss = Conv2D(1, kernel_size=(1, 1), padding="same")(ss)
    edge_out = Activation("sigmoid", name="edge_out")(ss)
    #######canny edge
    # canny = cv2.Canny(np.asarray(inputs),10,100)
    cat = Concatenate()([edge_out, canny])
    cw = Conv2D(1, kernel_size=(1, 1), padding="same")(cat)
    acts = Activation("sigmoid")(cw)
    edge = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same")(acts)
    edge = BatchNormalization()(edge)
    edge = Activation("relu")(edge)

    #########################

    n34_preinput = Attention_B(n33, n43, 128)
    n34 = Up3(n34_preinput, n43)
    n34_d = dual_att_blocks(n33, n43, 128)
    n34_t = Concatenate()([n34, n34_d])
    n34_t = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding="same")(n34_t)
    n34_2 = BatchNormalization()(n34_t)
    n34_2 = Activation("relu")(n34_2)
    n34_2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(n34_2)
    n34_2 = BatchNormalization()(n34_2)
    n34_2 = Activation("relu")(n34_2)
    n34_2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(n34_2)
    n34 = Add()([n34_2, n34_t])
    pred4 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="sigmoid")(n34)
    pred4 = UpSampling2D(size=(4, 4), interpolation="bilinear", name="pred4")(pred4)

    n24_preinput = Attention_B(n23, n34, 64)
    n24 = Up3(n24_preinput, n34)
    n24_d = dual_att_blocks(n23, n34, 64)
    n24_t = Concatenate()([n24, n24_d])
    n24_t = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding="same")(n24_t)
    n24_2 = BatchNormalization()(n24_t)
    n24_2 = Activation("relu")(n24_2)
    n24_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(n24_2)
    n24_2 = BatchNormalization()(n24_2)
    n24_2 = Activation("relu")(n24_2)
    n24_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(n24_2)
    n24 = Add()([n24_2, n24_t])
    pred2 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="sigmoid")(n24)
    pred2 = UpSampling2D(size=(2, 2), interpolation="bilinear", name="pred2")(pred2)

    n14_preinput = Conv2DTranspose(32, 4, strides=(2, 2), padding="same")(n24)
    n14_input = Concatenate()([n14_preinput, n13])
    n14_input = Conv2D(32, 1, activation="relu", padding="same", kernel_initializer="he_normal")(n14_input)
    n14 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n14_input)
    n14 = BatchNormalization()(n14)
    n14 = Add()([n14, n14_input])
    n14 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(n14)
    x = Conv2D(1, (1, 1), strides=(1, 1), padding="same", activation="sigmoid", name="x")(n14)
    model = Model(inputs=[inputs_img, att], outputs=[x, pred2, pred4, pred8])
    return model


def RDDB(x, y, nf1=128, nf2=1212, gc=64, bias=True):
    x1 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(x)
    x1 = LeakyReLU(alpha=0.25)(x1)

    y1 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(y)
    y1 = LeakyReLU(alpha=0.25)(y)

    x1c = Conv2D(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(x)
    x1c = LeakyReLU(alpha=0.25)(x1c)
    y1t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(y)
    y1t = LeakyReLU(alpha=0.25)(y1t)

    x2_input = concatenate([x, x1, y1t], axis=-1)
    x2 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(x2_input)
    x2 = LeakyReLU(alpha=0.25)(x2)

    y2_input = concatenate([y, y1, x1c], axis=-1)
    y2 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(y2_input)
    y2 = LeakyReLU(alpha=0.25)(y2)

    x2c = Conv2D(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(x1)
    x2c = LeakyReLU(alpha=0.25)(x2c)
    y2t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(y1)
    y2t = LeakyReLU(alpha=0.25)(y2t)

    x3_input = concatenate([x, x1, x2, y2t], axis=-1)
    x3 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(x3_input)
    x3 = LeakyReLU(alpha=0.25)(x3)

    y3_input = concatenate([y, y1, y2, x2c], axis=-1)
    y3 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(y3_input)
    y3 = LeakyReLU(alpha=0.25)(y3)

    x3c = Conv2D(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(x3)
    x3c = LeakyReLU(alpha=0.25)(x3c)
    y3t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(y3)
    y3t = LeakyReLU(alpha=0.25)(y3t)

    x4_input = concatenate([x, x1, x2, x3, y3t], axis=-1)
    x4 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(x4_input)
    x4 = LeakyReLU(alpha=0.25)(x4)

    y4_input = concatenate([y, y1, y2, y3, x3c], axis=-1)
    y4 = Conv2D(filters=gc, kernel_size=3, strides=1, padding="same", bias=bias)(y4_input)
    y4 = LeakyReLU(alpha=0.25)(y4)

    x4c = Conv2D(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(x4)
    x4c = LeakyReLU(alpha=0.25)(x4c)
    y4t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2, padding="same", bias=bias)(y4)
    y4t = LeakyReLU(alpha=0.25)(y4t)

    x5_input = concatenate([x, x1, x2, x3, x4, y4t], axis=-1)
    x5 = Conv2D(filters=nf1, kernel_size=3, strides=1, padding="same", bias=bias)(x5_input)
    x5 = LeakyReLU(alpha=0.25)(x5)

    y5_input = concatenate([y, y1, y2, y3, y4, x4c], axis=-1)
    y5 = Conv2D(filters=nf2, kernel_size=3, strides=1, padding="same", bias=bias)(y5_input)
    y5 = LeakyReLU(alpha=0.25)(y5)

    x5 = Lambda(lambda x: x * 0.4)(x5)
    y5 = Lambda(lambda x: x * 0.4)(y5)

    return Add()([x5, x]), Add()([y5, y])


G = sau()
G.summary()
checkpoint = ModelCheckpoint(
    "unet_connected_moreRRDB.hdf5",
    monitor="val_dice_coef",
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
    mode="max",
    period=1,
)
from keras.models import load_model


def seg_loss(y_true, y_pred):
    dice_s = dice_coefficient_loss(y_true, y_pred)

    # ce_loss = BinaryCrossentropy(y_true,y_pred)
    ce_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)

    return ce_loss + dice_s


def el(y_true, y_pred):
    l = keras.losses.BinaryCrossentropy(y_true, y_pred)
    return l


def get_optimizer():

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam


# G.load_weights("best_model_4_level.hdf5")
# G.compile(optimizer = Adam(lr = 1e-4), loss = dice_coefficient_loss, metrics = ['accuracy',"binary_crossentropy",dice_coef])
# G= load_model('best_model.h5')
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
        return 1
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin))


def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.0
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
            mean_dice_channel += channel_dice / (channel_num * batch_size)
    return mean_dice_channel


def train(epochs, batch_size, output_dir, model_save_dir):

    batch_count = int(len(train_x) / batch_size)
    max_val_dice = -1
    G = sau()
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
            X_tot = [get_image(sample_file, 288, 384) for sample_file in X_batch_list]
            X_batch, edge_x = [], []
            for i in range(0, batch_size):
                X_batch.append(X_tot[i][0])
                edge_x.append(X_tot[i][1])
            X_batch = np.array(X_batch).astype(np.float32)
            edge_x = np.array(edge_x).astype(np.float32)
            Y_tot = [get_image(sample_file, 288, 384, gray=True) for sample_file in Y_batch_list]
            Y_batch, edge_y = [], []
            for i in range(0, batch_size):
                Y_batch.append(Y_tot[i][0])
                edge_y.append(Y_tot[i][1])
            Y_batch = np.array(Y_batch).astype(np.float32)
            edge_y = np.array(edge_y).astype(np.float32)
            Y_batch = np.expand_dims(Y_batch, axis=3)
            edge_y = np.expand_dims(edge_y, axis=3)
            edge_x = np.expand_dims(edge_x, axis=3)
            """for i in range(0,batch_size):
                edge_batch.append(cv2.Canny(np.asarray(np.uint8(Y_batch[i])),10,100))
            edge_batch = np.asarray(edge_batch)
            edge_batch  =  np.expand_dims(edge_batch,axis=3)

            for i in range(0,batch_size):
                input_edge.append(cv2.Canny(np.asarray(np.uint8(X_batch[i])),10,100))
            input_edge = np.asarray(input_edge) 
            input_edge =np.expand_dims(input_edge,axis=3)"""
            # print(Y_batch.shape)
            G.train_on_batch([X_batch, edge_x], [Y_batch, edge_y, Y_batch, Y_batch])

        y_pred, _, _, _ = G.predict([X_val, edge_x_val], batch_size=5)
        y_pred = (y_pred >= 0.5).astype(int)
        res = mean_dice_coef(Y_val, y_pred)
        if res > max_val_dice:
            max_val_dice = res
            G.save("isic_ws.h5")
            print("New Val_Dice HighScore", res)


model_save_dir = "./model/"
output_dir = "./output/"
train(125, 4, output_dir, model_save_dir)
test_img_list = glob("../data/isic/test/image/*.jpg")
test_mask_list = glob("../data/isic/test/mask/*.jpg")
print(test_img_list)
print(test_mask_list)
G = sau()
G.load_weights("isic_ws.h5")
G.summary()
optimizer = get_optimizer()
G.compile(
    optimizer=optimizer,
    loss={"x": seg_loss, "edge_out": "binary_crossentropy", "pred4": seg_loss, "pred2": seg_loss},
    loss_weights={"x": 1.0, "edge_out": 1.0, "pred4": 1.0, "pred2": 1.0},
)

X_tot_test = [get_image(sample_file, 384, 288) for sample_file in test_img_list]
X_test, edge_x_test = [], []
for i in range(0, len(test_img_list)):
    X_test.append(X_tot_test[i][0])
    edge_x_test.append(X_tot_test[i][1])
X_test = np.array(X_test).astype(np.float32)
edge_x_test = np.array(edge_x_test).astype(np.float32)
print(edge_x_test.shape)
edge_x_test = np.expand_dims(edge_x_test, axis=3)
Y_tot_test = [get_image(sample_file, 384, 288, gray=True) for sample_file in test_mask_list]
Y_test, edge_y_test = [], []
for i in range(0, len(test_img_list)):
    Y_test.append(Y_tot_test[i][0])
Y_test = np.array(Y_test).astype(np.float32)

Y_test = np.expand_dims(Y_test, axis=3)


y_pred, _, _, _ = G.predict([X_test, edge_x_test], batch_size=5)
y_pred = (y_pred >= 0.5).astype(int)
res = mean_dice_coef(Y_test, y_pred)
np.save("X_test_isic.npy", X_test)
np.save("Y_test_isic.npy", Y_test)
np.save("Y_pred_isic.npy", y_pred)
print("dice coef on test set", res)


def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()
    # intersection = np.sum(intersection)
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp/(tp+fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (intersection + 1e-15) / (union + 1e-15), precision, recall


res = compute_iou(y_pred, Y_test)
print("iou on test set is ", res[0], " precision is ", res[1], " recall is ", res[2])
y_pred, _, _, _ = G.predict([X_val, edge_x_val], batch_size=5)
y_pred = (y_pred >= 0.5).astype(int)
res = mean_dice_coef(Y_val, y_pred)

print("dice coef on our val set", res)
