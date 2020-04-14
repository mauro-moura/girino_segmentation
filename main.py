
from unet import unet_completa, dice_coef
from utils import create_folder, normalize, load_images
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import io
import glob

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
size_img = 160

x_train = sorted(glob.glob('./dados_girino/Train/*'))
y_train = sorted(glob.glob('./dados_girino/GT/*'))

images = load_images(x_train)
masks = load_images(y_train)

image_generator = image_datagen.flow(images, masks,
    batch_size=8,
    seed=seed)

model = unet_completa(size_img, seed)
model.fit_generator(
    image_generator,
    steps_per_epoch=220,
    epochs=50)

model.save('girino_test.h5')

print("Carregando novas imagens")
new_imgs = sorted(glob.glob('./dados_girino/MeanAnisoImJ/*'))
new_imgs_load = load_images(new_imgs)

''' Realizando Novas Predições '''
new_predicao = model.predict(new_imgs_load)
new_predicao = new_predicao > 0.5
new_predicao = np.float64(new_predicao)
create_folder('outputs')
for i in range(len(new_predicao)):
    io.imsave('./outputs/predicao_%i.png'%(i), new_predicao[i])

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
