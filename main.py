
from unet import unet_completa, dice_coef
from utils import create_folder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import glob

data_gen_args = dict(rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
size_img = 160

x_train = sorted(glob.glob('./dados_girino/Train/*'))
y_train = sorted(glob.glob('./dados_girino/GT/*'))

from skimage import io

images = []
masks = []
for i in range(len(x_train)):
    images.append(io.imread(x_train[i]))
    masks.append(io.imread(y_train[i]))

import numpy as np
images = np.asarray(images)
images = images.reshape(-1, 160, 160, 1)
masks = np.asarray(masks)
masks = masks.reshape(-1, 160, 160, 1)

create_folder('./dados_girino/Aug_Train')
create_folder('./dados_girino/Aug_GT')

image_datagen.fit(images, augment=True, seed=seed)
image_generator = image_datagen.flow(images,
    save_to_dir='./dados_girino/Aug_Train/',
    save_prefix='N',
    save_format='tiff',
    batch_size=2,
    seed=seed)

mask_datagen.fit(images, augment=True, seed=seed)
mask_generator = mask_datagen.flow(masks,
    save_to_dir='./dados_girino/Aug_GT/',
    save_prefix='N',
    save_format='tiff',
    batch_size=2,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model = unet_completa(size_img)
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=1)

model.save('girino_test.h5')

predicao = model.predict(images)
predicao = predicao > 0.5
predicao = np.float64(predicao)
masks = np.fload64(masks)

sess = tf.InteractiveSession()
_dice = dice_coef(predicao, masks).eval()
sess.close()
np.savetxt('dice_metric.txt', _dice)

create_folder('outputs')
for i in range(len(predicao)):
    io.imsave('./outputs/predicao_%i.png'%(i), predicao[i])

print(_dice)
