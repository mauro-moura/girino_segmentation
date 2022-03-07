
from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
from skimage import io

import glob
import time

from unet import unet_completa, unet_mini, dice_coef
from utils import create_folder, load_images, load_images_array, reverse_size
from sklearn.model_selection import train_test_split

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
SEED = int(time.time())
ORIGINAL_SIZE = 850
NEW_SIZE = 320

norm_imgs = sorted(glob.glob('./dados_girino/A1_norm_images/*'))
GT_imgs = sorted(glob.glob('./dados_girino/A2_GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X = load_images_array(norm_imgs, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)
Y = load_images_array(GT_imgs, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

use_batch_size = 8
image_generator = image_datagen.flow(X_train, Y_train,
    batch_size=use_batch_size,
    seed=SEED)

# validation_generator = image_datagen.flow(X_train, Y_train,
#     batch_size=8,
#     seed=SEED)

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
callback = [es]

time_train_1 = time.time()

model = unet_completa(NEW_SIZE, SEED)

epoch = 50
model.fit_generator(
    image_generator,
    steps_per_epoch=385,
    epochs=epoch,
    callbacks=callback,
    )

#validation_data=validation_generator,
#validation_steps=38

time_train_2 = time.time()
TEMPO.append(time_train_2 - time_train_1)

name_file = str(SEED) + "_" + str(use_batch_size) + "_" + str(epoch)
model.save('girino_%s.h5'%name_file)

create_folder('outputs')

print("Calculando o dice para as imagens de teste")

time_test_1 = time.time()

predicao = model.predict(X_test)
predicao = predicao > 0.5
predicao = np.float64(predicao)
dice_metric = []
for i in range(len(predicao)):
    sess = tf.InteractiveSession()
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

with open('./outputs/tempos%s.txt'%name_file, 'w') as file:
    file.write(str(d))
