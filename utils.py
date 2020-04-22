
import os
import numpy as np
from skimage import io
import cv2
import glob

def load_images(img_list, size_img = 160, new_size = None):
    '''
    Recebe um glob das imagens e converte em um numpy array no formato que o Keras aceita
    '''
    img = []
    for i in range(len(img_list)):
        img.append(io.imread(img_list[i]))
    #img = np.asarray(img)
    img = np.float64(img)
    img = normalize(img)
    if (new_size != None):
        img = resize_img(img, new_size, new_size)
    else:
        img = get_shape_resize(img, size_img)
    img = img.reshape(-1, img.shape[-2], img.shape[-1], 1)
    return img

def get_shape_resize(img, size):
    '''
    Encontra o próximo tamanho da imagem se não for múltiplo de 8 e faz um resize
    '''
    if (size%8 != 0):
        new_shape = round(img.shape[-1]/8) * 8
        print('Um resize pode ser feito para %ix%i'%(new_shape, new_shape))
        new_img = resize_img(img, new_shape, new_shape)
        return np.float64(new_img)
    else:
        return img

def resize_img(img, width, height):
    resized = []
    for i in range(len(img)):
        resized.append(cv2.resize(img[i], (width, height)))
    return np.float64(resized)

def reverse_size(img, new_size):
    curr_img = cv2.resize(img, (new_size, new_size))
    curr_img = np.reshape(curr_img, (new_size, new_size, 1))
    return curr_img

def create_folder(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

def normalize(images):
    m = np.max(images)
    mi = np.min(images)
    if (m != mi):
        images = (images - mi) / (m - mi)
    return images
