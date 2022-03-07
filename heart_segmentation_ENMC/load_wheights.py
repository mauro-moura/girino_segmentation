# -*- coding: utf-8 -*-

from unet import dice_coef, dice_coef_loss
#from utils import get_images, separate_images, roi, resize_img, normalize, reshape_images
from utils import get_images, make_vol, image_preprocess, separate_train_test
import glob


x_train = glob.glob('data_heart/imagesTr/*')
y_train = glob.glob('data_heart/labelsTr/*')
img_xtrain, index = get_images(x_train,None,None)
img_ytrain, index = get_images(y_train,None,None)

num = 17
size_img = 128

train_X, valid_X, index_train_x, index_test_x = separate_train_test(img_xtrain, index = num)
train_ground, valid_ground, index_train_y, index_test_y = separate_train_test(img_ytrain, index = num)

train_X, valid_X, train_ground, valid_ground = image_preprocess(train_X, valid_X, train_ground, valid_ground)

'''
x_train = glob.glob('data_heart/imagesTr/*')
y_train = glob.glob('data_heart/labelsTr/*')
img_xtrain, index = get_images(x_train,None,None)
img_ytrain, index = get_images(y_train,None,None)
train_X, valid_X = separate_images(img_xtrain)
train_ground, valid_ground = separate_images(img_ytrain)

w = 320
x1 = 90
x2 = 218
size_img = 128

train_X, valid_X = roi(train_X, x1, x2, w, w), roi(valid_X, x1, x2, w, w)
train_ground, valid_ground = roi(train_ground, x1, x2, w, w), roi(valid_ground, x1, x2, w, w)

train_X, valid_X = resize_img(train_X, size_img, size_img), resize_img(valid_X, size_img, size_img)
train_ground, valid_ground = resize_img(train_ground, size_img, size_img), resize_img(valid_ground, size_img, size_img)

train_X, valid_X = normalize(train_X), normalize(valid_X)
train_ground, valid_ground = normalize(train_ground), normalize(valid_ground)

train_X, valid_X = reshape_images(train_X, size_img,size_img), reshape_images(valid_X, size_img,size_img)
train_ground, valid_ground = reshape_images(train_ground, size_img,size_img), reshape_images(valid_ground, size_img,size_img)
'''

from keras.models import load_model
model = load_model('Heart_Model.h5', custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef' : dice_coef})

'''
import numpy as np
import tensorflow as tf

predicao = model.predict(valid_X)
predicao = predicao > 0.5
predicao = np.float64(predicao)

predicao1 = model.predict(train_X)
predicao = predicao > 0.5
predicao = np.float64(predicao)

dice = dice_coef(predicao, valid_ground)
dice1 = dice_coef(predicao1, valid_X)
sess = tf.InteractiveSession()
dice = dice.eval()
sess.close()

plot_images(valid_X,size_img,size_img,70, save = False)

plot_images(predicao,size_img,size_img,10)

plot_images(valid_ground,size_img,size_img,10, save = True)

unir_imagem(valid_X, valid_ground,70, size_img)
unir_imagem(valid_X, predicao, 70, size_img)


'''
