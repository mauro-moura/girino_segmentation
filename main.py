
from unet import unet_completa, dice_coef
from utils import create_folder, normalize
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import io
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

images = []
masks = []
for i in range(len(x_train)):
    images.append(io.imread(x_train[i]))
    masks.append(io.imread(y_train[i]))

images = np.asarray(images)
images = images.reshape(-1, 160, 160, 1)
images = np.float64(images)
masks = np.asarray(masks)
masks = masks.reshape(-1, 160, 160, 1)
masks = np.float64(masks)

create_folder('./dados_girino/Aug_Train')
create_folder('./dados_girino/Aug_GT')

#image_datagen.fit(images, augment=True, seed=seed)
image_generator = image_datagen.flow(images, masks,
    batch_size=32,
    seed=seed)
'''
image_generator = image_datagen.flow(images,
    save_to_dir='./dados_girino/Aug_Train/',
    save_prefix='N',
    save_format='tif',
    batch_size=32,
    seed=seed)
'''
'''
mask_datagen.fit(images, augment=True, seed=seed)
mask_generator = mask_datagen.flow(masks,
    batch_size=32,
    seed=seed)
'''
'''
mask_generator = mask_datagen.flow(masks,
    save_to_dir='./dados_girino/Aug_GT/',
    save_prefix='N',
    save_format='tif',
    batch_size=32,
    seed=seed)
'''
# combine generators into one which yields image and masks
#train_generator = zip(image_generator, mask_generator)

model = unet_completa(size_img)

model.fit_generator(
    image_generator,
    steps_per_epoch=300,
    epochs=20)

#model.save('girino_test.h5')

images = normalize(images)
masks = normalize(masks)

print("Carregando novas imagens")
new_imgs = sorted(glob.glob('./dados_girino/MeanAnisoImJ/*'))
new_imgs_load = []
for i in range(len(new_imgs)):
    new_imgs_load.append(io.imread(new_imgs[i]))
new_imgs_load = np.asarray(new_imgs_load)
new_imgs_load = new_imgs_load.reshape(-1, 160, 160, 1)
new_imgs_load = normalize(new_imgs_load)
new_predicao = model.predict(new_imgs_load)
new_predicao = new_predicao > 0.5
new_predicao = np.float64(new_predicao)

for i in range(len(new_predicao)):
    io.imsave('./outputs/predicao_%i.png'%(i), new_predicao[i])

print("Calculando o dice para as máscaras conhecidas")

predicao = model.predict(images)
predicao = predicao > 0.5
predicao = np.float64(predicao)

sess = tf.InteractiveSession()
dice_metric = []
dice_metric.append(dice_coef(predicao, masks).eval())
sess.close()

print(dice_metric)

create_folder('outputs')
with open('./outputs/dice_metric.txt', 'w') as file:
    file.write(str(dice_metric))




