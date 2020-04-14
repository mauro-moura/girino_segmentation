
import os
import numpy as np
from skimage import io
import glob

def load_images(img_list, size_img = 160):
    '''
    Recebe um glob das imagens e converte em um numpy array no formato que o Keras aceita
    '''
    img = []
    for i in range(len(img_list)):
        img.append(io.imread(img_list[i]))
    img = np.asarray(img)
    img = img.reshape(-1, size_img, size_img, 1)
    img = np.float64(img)
    img = normalize(img)
    return img

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
