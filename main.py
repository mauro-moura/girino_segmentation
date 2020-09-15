
import glob
import time
import tensorflow as tf
import numpy as np
from skimage import io
from modules.Models.Unet import unet_completa
from modules.Models.Vnet import vnet
from modules.Models.LstmUnet import BCDU_net_D3
from modules.data_augmentation import MauroDataGenerator
from modules.utils import dice_coef, dice_coef_loss
from modules.utils import create_folder, load_images, reverse_size

g_time1 = time.time()

TEMPO = []
SEED = 1
ORIGINAL_SIZE = 850 #Antigo Size Img
NEW_SIZE = 128 #Tamanho para qual as imagens serão convertidas

#Imagens base
x_train = sorted(glob.glob('./dados_girino/Train/*.tif'))
y_train = sorted(glob.glob('./dados_girino/GT/*.tif'))
images = load_images(x_train, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)
masks = load_images(y_train, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)

'''
# Imagens Augmentadas
x_entrada = sorted(glob.glob('./dados_girino/Aug_Train/*.tiff'))
y_entrada = sorted(glob.glob('./dados_girino/Aug_GT/*.tiff'))
IMG_ENTRADA = load_images(x_entrada, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)
IMG_SAIDA = load_images(y_entrada, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE)
'''

dataAug = MauroDataGenerator()
IMG_ENTRADA, IMG_SAIDA = dataAug.run_all(x_train, y_train,
                                         images, masks,
                                         size_img=NEW_SIZE,
                                         n_img=30,
                                         Aug_2D=False)

time1 = time.time()
#model = unet_completa(NEW_SIZE, SEED, metric_loss= dice_coef_loss, metric= dice_coef)
#model = BCDU_net_D3(input_size = (NEW_SIZE, NEW_SIZE, 1), metric_loss= dice_coef_loss, metric= dice_coef)
model = vnet(input_size = (20, NEW_SIZE, NEW_SIZE, 1), loss = dice_coef_loss, metrics = [dice_coef])

model.fit(IMG_ENTRADA, IMG_SAIDA, batch_size=2, epochs=50)

time2 = time.time()
TEMPO.append(time2 - time1)

#model.save('girino_test.h5')
print('-------------------------------------------------')
print("Carregando novas imagens")
print('-------------------------------------------------')
# Imagens que desejamos verificar (Girino completo)
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

print('-------------------------------------------------')
print("Predizendo " + str(len(new_predicao)) + " Imagens")
print('-------------------------------------------------')

create_folder('outputs')
for i in range(len(new_predicao)):
    io.imsave('./outputs/predicao_%s.png'%(new_imgs[i][-8:-4]), reverse_size(new_predicao[i], new_size = ORIGINAL_SIZE))

print('-------------------------------------------------')
print("Calculando o dice para as máscaras conhecidas")
print('-------------------------------------------------')

predicao = model.predict(images)
predicao = predicao > 0.5
predicao = np.float64(predicao)
dice_metric = []
for i in range(len(predicao)):
    sess = tf.InteractiveSession()
    dice_metric.append(dice_coef(predicao[i], masks[i]).eval())
    sess.close()

print('-------------------------------------------------')
print('Salvando valores de Dice...\nMédia dos Dices: ' + str(np.mean(dice_metric)))
print('-------------------------------------------------')

with open('./outputs/dice_metric.txt', 'w') as file:
    file.write(str(dice_metric))

print('-------------------------------------------------')
print('Calculando Tempo')
print('-------------------------------------------------')

g_time2 = time.time()
TEMPO.append(g_time2 - g_time1)
d = {'Tempo do modelo': TEMPO[0],
     'Tempo de predição': TEMPO[1],
     'Tempo total': TEMPO[2]}
with open('./outputs/tempos.txt', 'w') as file:
    file.write(str(d))
