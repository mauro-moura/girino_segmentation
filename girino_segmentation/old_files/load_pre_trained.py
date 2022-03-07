
import glob

import tensorflow as tf
import numpy as np
from skimage import io

from unet import unet_completa, unet_mini, dice_coef
from utils import create_folder, load_images, reverse_size, resize_one_img, load_images_array, load_images_array_return_shape

ORIGINAL_SIZE = 850 #Antigo Size Img
NEW_SIZE = 320 #Tamanho para qual as imagens serão convertidas, deixe igual ao original se não for alterar

batch = [4, 8]
filename = ['girino_1627047982_4', 'girino_1627054868_8_50']

index = 0

from tensorflow import keras
model = keras.models.load_model('./pesos/%s.h5'%(filename[index]), compile=False)

print("Carregando novas imagens")
new_imgs = sorted(glob.glob('./dados_girino/Producao/*'))
new_imgs_load , img_shape = load_images_array_return_shape(new_imgs, ORIGINAL_SIZE, NEW_SIZE)

GT_Test = sorted(glob.glob('./dados_girino/GT_Producao/*'))
GT_Test_dice = load_images_array(GT_Test, new_size = NEW_SIZE)

new_predicao = model.predict(new_imgs_load)
new_predicao = new_predicao > 0.5
new_predicao = np.float64(new_predicao)

print("Predizendo " + str(len(new_predicao)) + " Imagens")
create_folder('outputs_prod')
for i in range(len(new_predicao)):
    io.imsave('./outputs_prod/predicao_%s_%s.png'%(str(i), str(batch[index])), resize_one_img(new_predicao[i], img_shape[1], img_shape[0])) # Usar o shape do new_images_load - 02.05.2021
    # grava em 850x850 mas nao está alterando a que vai para o dice para comparacao

print("Calculando o dice de produção")

dice_metric = []
sess = tf.InteractiveSession()
for i in range(len(new_predicao)):
    print(i)
    dice_metric.append(dice_coef(new_predicao[i], GT_Test_dice[i]).eval())
sess.close()

print('Salvando valores de Dice...\nMédia dos Dices: ' + str(np.mean(dice_metric)))
with open('./outputs_prod/dice_metric_production_%s.txt'%(str(batch[index])), 'w') as file:
    file.write(str(dice_metric))

