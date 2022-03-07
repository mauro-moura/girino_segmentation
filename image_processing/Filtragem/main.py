
import glob
import cv2
import os 

from PIL import Image
from skimage import io

import numpy as np

def load(img_list):
    img = []
    for i in range(len(img_list)):
        img.append(io.imread(img_list[i]))
    return img

def create_folder(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Diretorio " , dirName ,  " Criado ")
    else:    
        print("Diretorio " , dirName ,  " ja existe")

def save_img(img, folder, filename, start_num = 319):
    for i in range(len(img)):
        print('Salvando imagem: %i'%(i))
        # Alterar o %03d para outros valores
        io.imsave('./%s/%s_%s.png'%(folder, filename, str("%04d" %(start_num + i))), img[i])

def mean_filter(image, filter_size):
    new_image = cv2.blur(image,(filter_size, filter_size))
    return new_image

def gaussian_filter(image, filter_size):
    new_image = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    return new_image

def median_filter(image, filter_size):
    new_image = cv2.medianBlur(image, filter_size)
    return new_image

# A1_norm_images ; Producao
folder_name = './A1_norm_images'
imgs = sorted(glob.glob('%s/*'%(folder_name)))
imgs_load = load(imgs)
imgs_load = np.float32(imgs_load)

filter_size = 9
imgs_list = []
for img in imgs_load:
    imgs_list.append(mean_filter(img, filter_size))

filter_type = folder_name + '_' + 'mean_filtered'
create_folder(filter_type)

# Norm_imgs : 25 ; Prod : 319
save_img(imgs_list, filter_type, 'norm_img', 0)
