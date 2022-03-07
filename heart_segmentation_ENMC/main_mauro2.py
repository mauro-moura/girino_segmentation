
import glob
import random
import cv2

from time import time
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from unet import unet_reduzida, unet_completa, unet_n, dice_coef
from unet_hedden import unet_hedden
from utils import get_images, separate_train_test, make_vol, image_preprocess, create_folder

import matplotlib.pyplot as plt

import bot
my_bot = bot.Bot()

hd = [45 , 25, 40 , 40 , 13, 40, 42 , 33, 6 , 43 , 37 , 8 , 28, 18, 46 , 37, 41 , 41 , 33, 33 ]
hu = [112, 95, 113, 116, 91, 98, 109, 98, 67, 105, 104, 81, 97, 88, 103, 90, 110, 113, 92, 101]

x_train = sorted(glob.glob('data_heart/imagesTr/*'))
y_train = sorted(glob.glob('data_heart/labelsTr/*'))
img_xtrain, index_x = get_images(x_train,hd,hu)
img_ytrain, index_y = get_images(y_train,hd,hu)

#predict_vol = []
dice_metric = []
tempo = []
local = 'heart_tests/heart_LOO_Hedden/'

create_folder(local)

use_batch_size = 4
epoch = 100
spe = 300

for i in range(len(img_xtrain)):
    random.seed(time())
    seed_min = 0
    seed_max = 2**20 # Foi diminuido para que a seed ficasse com 9 digitos (Bobagem minha)
    SEED_1 = random.randint(seed_min, seed_max)
    SEED_2 = random.randint(seed_min, seed_max)
    SEED_3 = random.randint(seed_min, seed_max)
    
    print("Rodando Pela %i vez"%(i+1))
    
    train_X, valid_X, index_train_x, index_test_x = separate_train_test(img_xtrain, index = i)
    train_ground, valid_ground, index_train_y, index_test_y = separate_train_test(img_ytrain, index = i)

    size_img = 128
    
    train_X, valid_X, train_ground, valid_ground = image_preprocess(train_X, valid_X, train_ground, valid_ground)
        
    inicial = time()
    #model = unet_n(size_img, mult = 2)
    model = unet_hedden(size_img, SEED_3)
    
    reduce = ReduceLROnPlateau(monitor = 'loss', patience = 2, verbose = 1)
    es = EarlyStopping(monitor = 'loss', patience = 5, verbose = 1)
    callback = [reduce, es]
    
    #history = model.fit_generator(image_generator, steps_per_epoch=spe, epochs=epoch, callbacks=callback, validation_data=valid_generator)
    history = model.fit(x = train_X, y = train_ground, batch_size = use_batch_size, epochs = epoch, callbacks = callback)
    final = time()
    
    model.save(local + 'Heart_Model_%i.h5'%(i))

    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(local + '/history_loss_%s.png'%(i))
    plt.close()
    np.save(local + '/my_history_%s.npy'%(i), history.history)
        
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

    try:
        my_bot.send_message("Teste %i finalizado em %s segundos"%(i+1, str(tempo_total))) # Isso é utilizado somente para o bot
    except:
        print("Algum erro ocorreu")
    
    K.clear_session()

np.savetxt(local + 'Dice Metric.txt', dice_metric)
np.savetxt(local + 'Tempo.txt', tempo)

try:
    my_bot.send_message("Bateria finalizada") # Isso é utilizado somente para o bot
except:
    print("Algum erro ocorreu")
