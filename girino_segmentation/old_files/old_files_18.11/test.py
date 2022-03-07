
from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

import sys
import glob
import time
import random

from unet import unet_completa, dice_coef
from utils import create_folder, load_images_array
from sklearn.model_selection import train_test_split

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
ORIGINAL_SIZE = 850
NEW_SIZE = 320

_folder = './dados_girino/Teste_3'

norm_imgs = sorted(glob.glob(_folder + '/A1_norm_images/*'))
GT_imgs = sorted(glob.glob(_folder + '/A2_GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X = load_images_array(norm_imgs, new_size = NEW_SIZE)
Y = load_images_array(GT_imgs, new_size = NEW_SIZE)

print("Maximo de X: ", np.max(X))
print("Minimo de X: ", np.min(X))
print("Maximo de Y: ", np.max(Y))
print("Minimo de Y: ", np.min(Y))

_tam = [67, 70, 73, 76, 78, 58, 67, 65, 61, 62, 67, 73, 69, 70, 57, 53, 69, 72, 59, 68]
_tam_now = 0
def leave_um_fora(list_orig, trim_a, trim_b):
    list1 = list_orig[:trim_a]
    list2 = list_orig[trim_b:]
    return np.concatenate((list1, list2), axis=0)

n_exec = 1

print(X.shape)

#X = leave_um_fora(X, _tam_now, _tam_now + _tam[n_exec - 1])
#Y = leave_um_fora(Y, _tam_now, _tam_now + _tam[n_exec - 1])

print(X.shape)
print(Y.shape)

print("Maximo de X: ", np.max(X))
print("Minimo de X: ", np.min(X))
print("Maximo de Y: ", np.max(Y))
print("Minimo de Y: ", np.min(Y))

#Y = Y > 0

plt.imshow(np.reshape(X[67], (320, 320)))
plt.show()
plt.imshow(np.reshape(Y[67], (320, 320)))
plt.show()
