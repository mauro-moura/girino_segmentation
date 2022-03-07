# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import keras.backend as K

from keras.models import load_model

from unet import dice_coef, dice_coef_loss
from utils import get_images, make_vol, image_preprocess, separate_train_test
import glob

import nibabel as nib

def save_nifti(vol, name = 'teste', index = 0):
    data = np.float64(vol[index])
    data = data.reshape(-1, size_img, size_img)
    import os
    img = nib.Nifti1Image(data, None)
    nib.save(img, os.path.join('Predicted_Vol', '%s.nii.gz'%(name)))

x_train = glob.glob('data_heart/imagesTr/*')
y_train = glob.glob('data_heart/labelsTr/*')
img_xtrain, index = get_images(x_train,None,None)
img_ytrain, index = get_images(y_train,None,None)

base_folder = './heart_tests/heart_LOO_Hedden/'
size_img = 128
dice_metric = []

for i in range(20):
    num = i
    print("Rodando pela %i vez"%(i))
    train_X, valid_X, index_train_x, index_test_x = separate_train_test(img_xtrain, index = num)
    train_ground, valid_ground, index_train_y, index_test_y = separate_train_test(img_ytrain, index = num)
    
    train_X, valid_X, train_ground, valid_ground = image_preprocess(train_X, valid_X, train_ground, valid_ground)
    
    #model = load_model('Models_LeaveoneOut/Heart_Model_%i.h5'%(num), custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef' : dice_coef})
    model = load_model(base_folder + 'Heart_Model_%i.h5'%(num), custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef' : dice_coef})
    
    predicao = model.predict(valid_X)
    predicao = predicao > 0.5
    predict_vol = make_vol(predicao, index_test_y)
    predicao = np.float64(predicao)
    
    sess = tf.InteractiveSession()
    dice_metric.append(dice_coef(predicao, valid_ground).eval())
    sess.close()

    print(dice_metric[i])
    
    K.clear_session()
    
    '''
    from data_analysis import plot_images, unir_imagem
    
    plot_images(valid_X,size_img,size_img,70, save = False)
    
    plot_images(predicao,size_img,size_img,10)
    
    plot_images(valid_ground,size_img,size_img,10, save = True)
    
    unir_imagem(valid_X, valid_ground,70, size_img)
    unir_imagem(valid_X, predicao, 70, size_img)
    '''

    #save_nifti(predict_vol, "imagem_%i"%(num))
    
    print("Acabou a Leitura")

np.savetxt(base_folder + 'Dice_Metric_prod_2.txt', dice_metric)

with open(base_folder + 'mean_median_results.txt', 'w') as f:
    f.write("Dices dos Volumes: " + str(dice_metric))
    f.write('\n')
    f.write("Media: " + str(np.mean(dice_metric)))
    f.write('\n')
    f.write("Mediana: " + str(np.median(dice_metric)))
    f.write('\n')
    f.write("Maximo: " + str(np.max(dice_metric)))
    f.write('\n')
    f.write("Minimo: " + str(np.min(dice_metric)))