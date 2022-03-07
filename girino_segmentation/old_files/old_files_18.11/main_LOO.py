
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
from unet_hedden2 import unet_hedden
from utils import create_folder, load_images_array
from sklearn.model_selection import train_test_split

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
ORIGINAL_SIZE = 850
NEW_SIZE = 256

_folder = './dados_girino/Teste_2'

norm_imgs = sorted(glob.glob(_folder + '/A1_norm_images/*'))
GT_imgs = sorted(glob.glob(_folder + '/A2_GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X_orig = load_images_array(norm_imgs, new_size = NEW_SIZE)
Y_orig = load_images_array(GT_imgs, new_size = NEW_SIZE)

print("Maximo: ", np.max(X_orig))
print("Minimo: ", np.min(X_orig))
print("Maximo: ", np.max(Y_orig))
print("Minimo: ", np.min(Y_orig))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
callback = [es]

use_batch_size = 4

epoch = 300
#spe = 385
spe = 300

# ********** SOMENTE PARA O CORAÇ˜AO **********
_tam = [67, 70, 73, 76, 78, 58, 67, 65, 61, 62, 67, 73, 69, 70, 57, 53, 69, 72, 59, 68]
_tam_now = 0
def leave_um_fora(list_orig, trim_a, trim_b):
    list1 = list_orig[:trim_a]
    list2 = list_orig[trim_b:]
    return np.concatenate((list1, list2), axis=0)
Y_orig = Y_orig > 0
# ********** SOMENTE PARA O CORAÇ˜AO **********

create_folder('outputs')

# Isso é utilizado somente para o bot
import bot
my_bot = bot.Bot()

n_exec = 1
n_fold = 5
#_tam_now = _tam[n_exec - 2]

for i in range(n_fold):
    # ********** SOMENTE PARA O CORAÇ˜AO **********
    X = leave_um_fora(X_orig, _tam_now, _tam_now + _tam[n_exec - 1])
    Y = leave_um_fora(Y_orig, _tam_now, _tam_now + _tam[n_exec - 1])
    Y = np.float64(Y)

    _tam_now += _tam[n_exec - 1]
    # ********** SOMENTE PARA O CORAÇ˜AO **********

    random.seed(time.time())
    seed_min = 0
    seed_max = 2**20 # Foi diminuido para que a seed ficasse com 9 digitos (Bobagem minha)
    SEED_1 = random.randint(seed_min, seed_max)
    SEED_2 = random.randint(seed_min, seed_max)
    SEED_3 = random.randint(seed_min, seed_max)

    #Seeds do teste 2, Exec 4
    #SEED_1 = 693043
    #SEED_2 = 73036
    #SEED_3 = 286296

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED_1)
    
    image_generator = image_datagen.flow(X_train, Y_train,
        batch_size=use_batch_size,
        seed=SEED_2)

    validation_generator = image_datagen.flow(X_test, Y_test,
        batch_size=use_batch_size,
        seed=SEED_2)
    time_train_1 = time.time()

    model = unet_completa(NEW_SIZE, SEED_3)
    #model = unet_hedden(NEW_SIZE, SEED_3)

    history = model.fit_generator(image_generator, steps_per_epoch=spe, epochs=epoch, callbacks=callback, validation_data=validation_generator)

    time_train_2 = time.time()
    TEMPO.append(time_train_2 - time_train_1)

    folder_name = './outputs/Exec_%s'%str(n_exec)
    create_folder(folder_name)
    name_file = str(use_batch_size) + "_" + str(epoch) + "_exec_%i"%n_exec
    model.save(folder_name + '/girino_%s.h5'%name_file)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(folder_name + '/history_loss_%s.png'%(name_file))
    plt.close()
    np.save(folder_name + '/my_history_%s.npy'%(name_file), history.history)

    print("Calculando o dice para as imagens de teste")

    time_test_1 = time.time()

    predicao = model.predict(X_test)
    predicao = predicao > 0.5
    predicao = np.float64(predicao)

    dice_metric = []
    sess = tf.InteractiveSession()
    for i in range(len(predicao)):
        dice_metric.append(dice_coef(predicao[i], Y_test[i]).eval())
    sess.close()

    time_test_2 = time.time()
    TEMPO.append(time_test_2 - time_test_1)

    print('Salvando valores de Dice...\nMédia dos Dices: ' + str(np.mean(dice_metric)))
    with open(folder_name + '/dice_metric_%s.txt'%name_file, 'w') as file:
        file.write(str(dice_metric))

    print('Calculando e gravando tempo')

    TEMPO.append(TEMPO[0] + TEMPO[1])

    d = {'Tempo de treinamento': TEMPO[0],
        'Tempo de teste': TEMPO[1],
        'Tempo total': TEMPO[2]}

    with open(folder_name + '/tempos_%s.txt'%name_file, 'w') as file:
        file.write(str(d))

    d_s = {'Seed do split': SEED_1,
        'Seed do Data Augmentation': SEED_2,
        'Seed dos pesos': SEED_3}

    with open(folder_name + '/seeds_%s.txt'%name_file, 'w') as file:
        file.write(str(d_s))
    
    K.clear_session()

    my_bot.send_message("Teste %i finalizado"%n_exec) # Isso é utilizado somente para o bot
    n_exec += 1

    time.sleep(60)

my_bot.send_message("Bateria finalizada") # Isso é utilizado somente para o bot
