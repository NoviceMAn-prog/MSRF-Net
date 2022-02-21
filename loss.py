import os
from glob import glob
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

np.random.seed(123)
import warnings

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, Nadam


def el(y_true, y_pred):
    l = keras.losses.BinaryCrossentropy(y_true, y_pred)
    return l


def get_optimizer():

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam


def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
        return 1
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, "float32")
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), "float32")
    intersection = y_true_f * y_pred_f
    score = 2.0 * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


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


def seg_loss(y_true, y_pred):
    dice_s = dice_coefficient_loss(y_true, y_pred)

    # ce_loss = BinaryCrossentropy(y_true,y_pred)
    ce_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)

    return ce_loss + dice_s


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
