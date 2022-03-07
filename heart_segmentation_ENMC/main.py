# -*- coding: utf-8 -*-
'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0" #model will be trained on GPU 1
'''

from unet import unet_reduzida, dice_coef
from utils import get_images, separate_images, roi, resize_img, normalize, reshape_images
import glob
from keras import backend as K
import numpy as np
import nibabel as nib #reading MR images
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

hd = [45 , 25, 40 , 40 , 13, 40, 42 , 33, 6 , 43 , 37 , 8 , 28, 18, 46 , 37, 41 , 41 , 33, 33 ]
hu = [112, 95, 113, 116, 91, 98, 109, 98, 67, 105, 104, 81, 97, 88, 103, 90, 110, 113, 92, 101]
######3####4###5####7###9###10###11##14##16###17###18###19##20##21##22###23##24###26###29##30##

x_train = glob.glob('data_heart/imagesTr/*')
y_train = glob.glob('data_heart/labelsTr/*')
img_xtrain, index_x = get_images(x_train,hd,hu)
img_ytrain, index_y = get_images(y_train,hd,hu)

train_X, valid_X = separate_images(img_xtrain)
train_ground, valid_ground = separate_images(img_ytrain)

del(x_train); del(y_train)

w = 320
x1 = 90
x2 = 218

size_img = 128

train_X = roi(train_X, x1, x2, w, w)
valid_X = roi(valid_X, x1, x2, w, w)

train_ground, valid_ground = roi(train_ground, x1, x2, w, w), roi(valid_ground, x1, x2, w, w)

train_X, valid_X = resize_img(train_X, size_img, size_img), resize_img(valid_X, size_img, size_img)
train_ground, valid_ground = resize_img(train_ground, size_img, size_img), resize_img(valid_ground, size_img, size_img)

train_X, valid_X = normalize(train_X), normalize(valid_X)
train_ground, valid_ground = normalize(train_ground), normalize(valid_ground)

train_X, valid_X = reshape_images(train_X, size_img,size_img), reshape_images(valid_X, size_img,size_img)
train_ground, valid_ground = reshape_images(train_ground, size_img,size_img), reshape_images(valid_ground, size_img,size_img)

del(w); del(x1); del(x2)
'''
model = unet_reduzida(size_img)

reduce = ReduceLROnPlateau(monitor = 'loss', patience = 2, verbose = 1)
es = EarlyStopping(monitor = 'loss', patience = 5, verbose = 1)

history = model.fit(train_X, train_ground, batch_size = 8, epochs = 50, callbacks = [reduce, es])

model.save('Heart_Model_3.h5')

predicao = model.predict(valid_X)
predicao = predicao > 0.5
predicao = np.float64(predicao)

dice = dice_coef(predicao, valid_ground)
sess = tf.InteractiveSession()
dice = dice.eval()

K.clear_session()
'''