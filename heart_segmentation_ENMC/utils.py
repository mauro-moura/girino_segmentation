
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import os
import numpy as np

def get_images(image,hd = None,hu = None):
    images = []
    index = []
    for f in range(len(image)):
        ai = []
        a = nib.load(image[f])
        a = a.get_data()
        if (hd != None and hu != None):
            a = a[:,:,hd[f]:hu[f]]
        for i in range(a.shape[2]):
            ai.append((a[:,:,i]))
        images.append(ai)
        index.append(len(ai))
    return images, index

def separate_train_test(image, index, split = 0.95):
    train = []
    test = []
    index_train = []
    index_test = []
    split = int(len(image) - (split * len(image)))
    inicio = index
    fim = index + split
    for k in range(len(image)):
        if k >= inicio and k < fim:
            for j in range(len(image[k])):
                test.append(image[k][j])
                index_test.append(k)
        else:
            for j in range(len(image[k])):
                train.append(image[k][j])
                index_train.append(k)
    return train, test, index_train, index_test

def separate_images(image, split = 0.9):
    train = []
    test = []
    split = int(split * len(image))
    for i in range(split):
        for j in range(len(image[i])):
            train.append(image[i][j])
    for i in range(split, len(image)):
        for j in range(len(image[i])):
            test.append(image[i][j])
    return train, test

def roi(img, x1, x2, w, h):
    imagem = []
    for i in range(len(img)):
        curr_img = np.reshape(img[i], (w,h))
        imagem.append(curr_img[x1:x2,x1:x2])
    return imagem

def resize_img(img, width, heigh):
    resized = []
    for i in range(len(img)):
        resized.append(cv2.resize(img[i], (width, heigh)))
    return resized

def reshape_images(images, width, height):
    images = np.asarray(images)
    images = images.reshape(-1, width, height, 1)
    return images

def normalize(images):
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi) / (m - mi)
    return images

def plot_images(images,width, height, index, save = False):
    curr_img = np.reshape(images[index], (width,height))
    plt.imshow(curr_img, cmap='gray')
    if save:
        plt.savefig('imagem.png')
    plt.show()
    
def unir_imagem(image1, image2, index, size_img):
    curr_img = np.reshape(image1[index], (size_img,size_img))
    curr_img2 = np.reshape(image2[index], (size_img,size_img))
    
    new_img = curr_img + curr_img2
    
    plt.imshow(new_img, cmap='gray')
    plt.show()
    
def image_preprocess(train_X, valid_X, train_ground, valid_ground, w = 320, x1 = 90, x2 = 218, size_img = 128):
    train_X, valid_X = roi(train_X, x1, x2, w, w), roi(valid_X, x1, x2, w, w)
    train_ground, valid_ground = roi(train_ground, x1, x2, w, w), roi(valid_ground, x1, x2, w, w)
   
    train_X, valid_X = resize_img(train_X, size_img, size_img), resize_img(valid_X, size_img, size_img)
    train_ground, valid_ground = resize_img(train_ground, size_img, size_img), resize_img(valid_ground, size_img, size_img)
    
    train_X, valid_X = normalize(train_X), normalize(valid_X)
    train_ground, valid_ground = normalize(train_ground), normalize(valid_ground)
    
    train_X, valid_X = reshape_images(train_X, size_img,size_img), reshape_images(valid_X, size_img,size_img)
    train_ground, valid_ground = reshape_images(train_ground, size_img,size_img), reshape_images(valid_ground, size_img,size_img)
    
    return train_X, valid_X, train_ground, valid_ground
    
def make_vol(image, index, width = 256, height = 256, reshape = False):
    '''
    
    Index no formato [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    '''
    vol = []
    a = []
    for i in range(len(index)):
        if (reshape):
            curr_img = image[i].reshape(width, height, -1)
        else:
            curr_img = image[i]
        #Condição de Adicionamento 
        if index[i-1] != index [i] and i != 0:
            vol.append(a)
            a = []
            a.append(curr_img)
        else:
            a.append(curr_img)
    vol.append(a)
    return vol

def create_folder(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Diretorio " , dirName ,  " Criado ")
    else:    
        print("Diretorio " , dirName ,  " ja existe")
