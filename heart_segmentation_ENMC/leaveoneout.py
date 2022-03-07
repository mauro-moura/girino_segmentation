# -*- coding: utf-8 -*-
'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0" #model will be trained on GPU 1
'''

from unet import unet_reduzida, unet_completa, unet_n, dice_coef
from utils import get_images, separate_train_test, make_vol, image_preprocess
import glob
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from time import time

hd = [45 , 25, 40 , 40 , 13, 40, 42 , 33, 6 , 43 , 37 , 8 , 28, 18, 46 , 37, 41 , 41 , 33, 33 ]
hu = [112, 95, 113, 116, 91, 98, 109, 98, 67, 105, 104, 81, 97, 88, 103, 90, 110, 113, 92, 101]

x_train = glob.glob('data_heart/imagesTr/*')
y_train = glob.glob('data_heart/labelsTr/*')
img_xtrain, index_x = get_images(x_train,hd,hu)
img_ytrain, index_y = get_images(y_train,hd,hu)

#predict_vol = []
dice_metric = []
tempo = []
local = 'leaveoneout_reduz/'

for i in range(len(img_xtrain)):
    
    print("Rodando Pela %i vez"%(i+1))
    
    train_X, valid_X, index_train_x, index_test_x = separate_train_test(img_xtrain, index = i)
    train_ground, valid_ground, index_train_y, index_test_y = separate_train_test(img_ytrain, index = i)
    
    size_img = 128
    
    train_X, valid_X, train_ground, valid_ground = image_preprocess(train_X, valid_X, train_ground, valid_ground)
    
    inicial = time()
    model = unet_n(size_img, mult = 4)
    
    reduce = ReduceLROnPlateau(monitor = 'loss', patience = 2, verbose = 1)
    es = EarlyStopping(monitor = 'loss', patience = 5, verbose = 1)
    
    history = model.fit(x = train_X, y = train_ground, batch_size = 8, epochs = 50, callbacks = [reduce, es])
    final = time()
    
    model.save(local + 'Heart_Model_%i.h5'%(i))
        
    predicao = model.predict(valid_X)
    #predict_vol.append(make_vol(predicao, index = index_test_y))
    predicao = predicao > 0.5
    predicao = np.float64(predicao)

    sess = tf.InteractiveSession()
    dice_metric.append(dice_coef(predicao, valid_ground).eval())
    sess.close()

    tempo_total = final - inicial
    tempo.append(tempo_total)
    print("rodou em : %f"%(tempo_total))
    K.clear_session()

np.savetxt(local + 'Dice Metric.txt', dice_metric)
np.savetxt(local + 'Tempo.txt', tempo)

