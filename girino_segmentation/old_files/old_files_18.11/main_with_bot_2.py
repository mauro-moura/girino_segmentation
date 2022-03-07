
from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob
import time
import random

from unet import unet_completa, unet_pura, dice_coef
from utils import create_folder, load_images_array
from sklearn.model_selection import train_test_split

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

'''
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
'''

image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
ORIGINAL_SIZE = 850
NEW_SIZE = 320

_folder = './dados_girino/Teste_2'
norm_imgs = sorted(glob.glob(_folder + '/A1_norm_images/*'))
GT_imgs = sorted(glob.glob(_folder + '/A2_GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X = load_images_array(norm_imgs, new_size = NEW_SIZE)
Y = load_images_array(GT_imgs, new_size = NEW_SIZE)

# Isso é utilizado somente para o bot
import bot
my_bot = bot.Bot()

n_exec = 1
n_fold = 2

random.seed(int(time.time()))
seed_max = 2**30
#SEED_1 = random.randint(0, seed_max)
#SEED_2 = random.randint(0, seed_max)
#SEED_3 = random.randint(0, seed_max)
SEED_1 = 33564
SEED_2 = 30010
SEED_3 = 154314

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED_1) # Antes tava com a seed 42, pode ter sido isso?

es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5)
callback = [es]

use_batch_size = 4
image_generator = image_datagen.flow(X_train, Y_train,
    batch_size=use_batch_size,
    seed=SEED_2)

time_train_1 = time.time()

model = unet_completa(NEW_SIZE, SEED_3)
#model = unet_pura(NEW_SIZE, SEED_3)

epoch = 50
#spe = 385
spe = 300

create_folder('outputs')

history = model.fit_generator(image_generator, steps_per_epoch=spe, epochs=epoch, callbacks=callback)

time_train_2 = time.time()
TEMPO.append(time_train_2 - time_train_1)

name_file = str(use_batch_size) + "_" + str(epoch)
model.save('./outputs/girino_%s.h5'%name_file)

plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('./outputs/history_loss.png')
plt.close()

np.save('./outputs/my_history.npy', history.history)

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
with open('./outputs/dice_metric_%s.txt'%name_file, 'w') as file:
    file.write(str(dice_metric))

print('Calculando e gravando tempo')

TEMPO.append(TEMPO[0] + TEMPO[1])

d = {'Tempo de treinamento': TEMPO[0],
     'Tempo de teste': TEMPO[1],
     'Tempo total': TEMPO[2]}

with open('./outputs/tempos_%s.txt'%name_file, 'w') as file:
    file.write(str(d))

d_s = {'Seed do split': SEED_1,
     'Seed do Data Augmentation': SEED_2,
     'Seed dos pesos': SEED_3}

with open('./outputs/seeds_%s.txt'%name_file, 'w') as file:
    file.write(str(d_s))

# Comente o que está abaixo para utilizar a versão sem bot
import shutil

output_filename = 'outputs'
dir_name = './outputs/'
shutil.make_archive(output_filename, 'zip', dir_name)

import bot

my_bot = bot.Bot()
my_bot.send_message("Tudo finalizado")