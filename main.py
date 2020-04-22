
from unet import unet_completa, unet_mini, dice_coef
from utils import create_folder, load_images, reverse_size
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import io
import glob
import time

g_time1 = time.time()
data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
SEED = 1
ORIGINAL_SIZE = 850 #Antigo Size Img
NEW_SIZE = 320 #Tamanho para qual as imagens serão convertidas, deixe igual ao original se não for alterar

x_train = sorted(glob.glob('./dados_girino/Train/*'))
y_train = sorted(glob.glob('./dados_girino/GT/*'))

for i in range(len(x_train)):
    if x_train[i][-8:-4] != y_train[i][-8:-4]:
        print('Algo está errado com as imagens')

images = load_images(x_train, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)
masks = load_images(y_train, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)

image_generator = image_datagen.flow(images, masks,
    batch_size=8,
    seed=SEED)

time1 = time.time()
model = unet_completa(NEW_SIZE, SEED)
model.fit_generator(
    image_generator,
    steps_per_epoch=220,
    epochs=50)
time2 = time.time()
TEMPO.append(time2 - time1)

#model.save('girino_test.h5')

print("Carregando novas imagens")
new_imgs = sorted(glob.glob('./dados_girino/MeanAnisoImJ/*'))
new_imgs_load = load_images(new_imgs, ORIGINAL_SIZE, NEW_SIZE)
''' 
#Realizando Novas Predições
'''
time1 = time.time()
new_predicao = model.predict(new_imgs_load)
new_predicao = new_predicao > 0.5
new_predicao = np.float64(new_predicao)
time2 = time.time()
TEMPO.append(time2 - time1)

print("Predizendo " + str(len(new_predicao)) + " Imagens")
create_folder('outputs')
for i in range(len(new_predicao)):
    io.imsave('./outputs/predicao_%s.png'%(new_imgs[i][-8:-4]), reverse_size(new_predicao[i], new_size = ORIGINAL_SIZE))

print("Calculando o dice para as máscaras conhecidas")
predicao = model.predict(images)
predicao = predicao > 0.5
predicao = np.float64(predicao)
dice_metric = []
for i in range(len(predicao)):
    sess = tf.InteractiveSession()
    dice_metric.append(dice_coef(predicao[i], masks[i]).eval())
    sess.close()
print('Salvando valores de Dice...\nMédia dos Dices: ' + str(np.mean(dice_metric)))
with open('./outputs/dice_metric.txt', 'w') as file:
    file.write(str(dice_metric))

print('Calculando Tempo')
g_time2 = time.time()
TEMPO.append(g_time2 - g_time1)
d = {'Tempo do modelo': TEMPO[0],
     'Tempo de predição': TEMPO[1],
     'Tempo total': TEMPO[2]}
with open('./outputs/tempos.txt', 'w') as file:
    file.write(str(d))

