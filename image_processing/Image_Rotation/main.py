
import glob

from PIL import Image
from skimage import io

import numpy as np

def load(img_list):
    img = []
    for i in range(len(img_list)):
        img.append(io.imread(img_list[i]))
    return img

def rotate(img, n_rotate = 2):
    new_img = []
    for i in range(len(img)):
        new_img.append(np.rot90(img[i], n_rotate))
    return new_img

def save_img(img, folder, filename, start_num = 319):
    for i in range(len(img)):
        print("Tratando Slice %i de %s"%(i, str(len(img))))
        io.imsave('./%s/%s_%s.png'%(folder, filename, str("%03d" %(start_num + i))), img[i])

def save_GT(img, folder, filename, start_num = 319):
    for i in range(len(img)):
        print("Tratando GT %i de %s"%(i, str(len(img))))
        im = Image.fromarray(img[i])
        im.save('./%s/%s_%s.tif'%(folder, filename, str("%03d" %(start_num + i))), 'TIFF')

name_1 = 'A1_norm_images'
name_2 = 'A2_GT_images'

print("Carregando Imagens")
new_imgs = sorted(glob.glob('./%s/*.png'%name_1))
new_imgs_load = load(new_imgs)

GT_Test = sorted(glob.glob('./%s/*.tif'%name_2))
GT_Test_dice = load(GT_Test)

new_imgs_load = rotate(new_imgs_load,2)
GT_Test_dice = rotate(GT_Test_dice,2)

#new_imgs_load = np.flip(new_imgs_load, axis=1)
#GT_Test_dice = np.flip(GT_Test_dice, axis=1)

print('Processando e salvando...')
save_img(new_imgs_load, '%s_2'%name_1, 'norm_image', 25)
save_GT(GT_Test_dice, '%s_2'%name_2, 'GT_bin', 25)
