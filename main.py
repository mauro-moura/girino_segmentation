
from unet import unet_completa, unet_mini, dice_coef
from utils import create_folder, load_images, load_images_array, reverse_size
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import io
import glob
import time
from sklearn.model_selection import train_test_split

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
SEED = 1
ORIGINAL_SIZE = 850 #Antigo Size Img
NEW_SIZE = 320 #Tamanho para qual as imagens serão convertidas, deixe igual ao original se não for alterar

norm_imgs = sorted(glob.glob('./dados_girino/A1_norm_images/*'))
GT_imgs = sorted(glob.glob('./dados_girino/A2_GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X = load_images_array(norm_imgs, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)
Y = load_images_array(GT_imgs, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

image_generator = image_datagen.flow(X_train, Y_train,
    batch_size=8,
    seed=SEED)

time_train_1 = time.time()

model = unet_completa(NEW_SIZE, SEED)

model.fit_generator(
    image_generator,
    steps_per_epoch=385,
    epochs=100)

time_train_2 = time.time()
TEMPO.append(time_train_2 - time_train_1)

model.save('girino_test.h5')


#Realizando Novas Predições - Imagens do conjunto de teste

create_folder('outputs') # Anderson 2021.05.02 - Aqui só vai ter a saída dos Dices. Deixar para salvar os resultados apenas para os conjuntos de produção - main_2.py

print("Calculando o dice para as imagens de teste") # Anderson 2021.05.02

time_test_1 = time.time() # Para o cálculo do tempo para o conjunto de teste

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
with open('./outputs/dice_metric.txt', 'w') as file:
    file.write(str(dice_metric))

#################################################################################
print('Calculando e gravando tempo')

# Cálculo e append do tempo total
TEMPO.append(TEMPO[0] + TEMPO[1])

d = {'Tempo de treinamento': TEMPO[0],
     'Tempo de teste': TEMPO[1],
     'Tempo total': TEMPO[2]}

with open('./outputs/tempos.txt', 'w') as file:
    file.write(str(d))
