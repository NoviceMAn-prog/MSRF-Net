import os
import re
import csv
import json
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import warnings
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D
from keras import backend as K
from keras.layers import concatenate ,Lambda
import itertools
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint 
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50,VGG19,VGG16,DenseNet121,DenseNet169,InceptionResNetV2
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
import numpy as np
from skimage.morphology import square,binary_erosion,binary_dilation,binary_opening,binary_closing
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras import regularizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from math import sqrt, ceil
from PIL import Image
import numpy as np
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from glob import glob
import tifffile as tif
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
from model import msrf
from model import *
import tensorflow as tf
from tensorflow.keras.callbacks import *
from keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def read_data(x, y):
    """ Read the image and mask from the given path. """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    return image, mask

def read_params():
    """ Reading the parameters from the JSON file."""
    with open("params.json", "r") as f:
        data = f.read()
        params = json.loads(data)
        return params

def load_data(path):
    """ Loading the data from the given path. """
    images_path = os.path.join(path, "image/*")
    masks_path  = os.path.join(path, "mask/*")

    images = glob(images_path)
    masks  = glob(masks_path)

    return images, masks

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y
def get_image(image_path, image_size_wight, image_size_height,gray=False):
    # load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
       
    if gray==True:
        img = img.convert('L')
    # center crop
    img_center_crop = img
    # resize
    img_resized = img
    edge = cv2.Canny(np.asarray(np.uint8(img_resized)),10,1000)
    
    flag = False
    # convert to numpy and normalize
    img_array = np.asarray(img_resized).astype(np.float32)/255.0
    edge = np.asarray(edge).astype(np.float32)/255.0
    #print(img_array)
    if gray==True:
        img_array=(img_array >=0.5).astype(int)
    img.close()
    return img_array,edge

from glob import glob


np.random.seed(42)
create_dir("files")

train_path = "data/kdsb/train/"
valid_path = "data/kdsb/val/"

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


print("final training set length",len(train_x),len(train_y))
print("final valid set length",len(valid_x),len(valid_y))

import random

    
X_tot_val = [get_image(sample_file,256,256) for sample_file in valid_x]
X_val,edge_x_val = [],[]
print(len(X_tot_val))
for i in range(0,len(valid_x)):
    X_val.append(X_tot_val[i][0])
    edge_x_val.append(X_tot_val[i][1])
X_val = np.array(X_val).astype(np.float32)
edge_x_val = np.array(edge_x_val).astype(np.float32)
edge_x_val  =  np.expand_dims(edge_x_val,axis=3)
Y_tot_val = [get_image(sample_file,256,256,gray=True) for sample_file in valid_y]
Y_val,edge_y = [],[]
for i in range(0,len(valid_y)):
    Y_val.append(Y_tot_val[i][0])
Y_val = np.array(Y_val).astype(np.float32)
           
Y_val  =  np.expand_dims(Y_val,axis=3)

def el(y_true, y_pred):
    l = keras.losses.BinaryCrossentropy(y_true,y_pred)
    return l
def get_optimizer():
 
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam
#G.load_weights("best_model_4_level.hdf5")
#G.compile(optimizer = Adam(lr = 1e-4), loss = dice_coefficient_loss, metrics = ['accuracy',"binary_crossentropy",dice_coef])
#G= load_model('best_model.h5')
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    return mean_dice_channel

def train(epochs, batch_size,output_dir, model_save_dir):
    
    batch_count = int(len(train_x) / batch_size)
    max_val_dice= -1
    G = sau()
    G.summary()
    optimizer = get_optimizer()
    G.compile(optimizer = optimizer, loss = {'x':seg_loss,'edge_out':'binary_crossentropy','pred4':seg_loss,'pred2':seg_loss},loss_weights={'x':2.,'edge_out':1.,'pred4':1. , 'pred2':1.})
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15,batch_size)
        #sp startpoint
        for sp in range(0,batch_count,1):
            if (sp+1)*batch_size>len(train_x):
                batch_end = len(train_x)
            else:
                batch_end = (sp+1)*batch_size
            X_batch_list = train_x[(sp*batch_size):batch_end]
            Y_batch_list = train_y[(sp*batch_size):batch_end]
            X_tot = [get_image(sample_file,256,256) for sample_file in X_batch_list]
            X_batch,edge_x = [],[]
            for i in range(0,batch_size):
                X_batch.append(X_tot[i][0])
                edge_x.append(X_tot[i][1])
            X_batch = np.array(X_batch).astype(np.float32)
            edge_x = np.array(edge_x).astype(np.float32)
            Y_tot = [get_image(sample_file,256,256, gray=True) for sample_file in Y_batch_list]
            Y_batch,edge_y = [],[]
            for i in range(0,batch_size):
                Y_batch.append(Y_tot[i][0])
                edge_y.append(Y_tot[i][1])
            Y_batch = np.array(Y_batch).astype(np.float32)
            edge_y = np.array(edge_y).astype(np.float32)
            Y_batch  =  np.expand_dims(Y_batch,axis=3)
            edge_y = np.expand_dims(edge_y,axis=3)
            edge_x = np.expand_dims(edge_x,axis=3)
            '''for i in range(0,batch_size):
                edge_batch.append(cv2.Canny(np.asarray(np.uint8(Y_batch[i])),10,100))
            edge_batch = np.asarray(edge_batch)
            edge_batch  =  np.expand_dims(edge_batch,axis=3)

            for i in range(0,batch_size):
                input_edge.append(cv2.Canny(np.asarray(np.uint8(X_batch[i])),10,100))
            input_edge = np.asarray(input_edge) 
            input_edge =np.expand_dims(input_edge,axis=3)'''
            #print(Y_batch.shape)
            G.train_on_batch([X_batch,edge_x],[Y_batch,edge_y,Y_batch,Y_batch])

        y_pred,_,_,_ = G.predict([X_val,edge_x_val],batch_size=5)
        y_pred = (y_pred >=0.5).astype(int)
        res = mean_dice_coef(Y_val,y_pred)
        if(res > max_val_dice):
            max_val_dice = res
            G.save('kdsb_ws.h5')
            print('New Val_Dice HighScore',res)            
            
model_save_dir = './model/'
output_dir = './output/'
train(125,4,output_dir,model_save_dir)