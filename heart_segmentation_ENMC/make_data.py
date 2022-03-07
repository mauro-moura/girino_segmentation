'''
Módulo para fazer a criação das imagens no formato de tiff
'''


#from unet import dice_coef, dice_coef_loss
from utils import get_images, make_vol, image_preprocess, separate_train_test
import glob
import imageio
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def save_nifti(vol, path = 'Predicted_Vol', name = 'teste', index = 0, reshape = False):
    data = np.float64(vol[index])
    if (reshape):
        data = data.reshape(-1, size_img, size_img)
    else:
        data = data.reshape(size_img, size_img, -1)
    import os
    img = nib.Nifti1Image(data, None)
    nib.save(img, os.path.join(path, '%s.nii.gz'%(name)))

hd = [45 , 25, 40 , 40 , 13, 40, 42 , 33, 6 , 43 , 37 , 8 , 28, 18, 46 , 37, 41 , 41 , 33, 33 ]
hu = [112, 95, 113, 116, 91, 98, 109, 98, 67, 105, 104, 81, 97, 88, 103, 90, 110, 113, 92, 101]

x_train = glob.glob('data_heart/imagesTr/*')
y_train = glob.glob('data_heart/labelsTr/*')
img_xtrain, index = get_images(x_train,hd,hu)
img_ytrain, index = get_images(y_train,hd,hu)

size_img = 256
#index_train = []
#index_test = []

i = 1
train_X, test_X, index_train, index_test = separate_train_test(img_xtrain, index = i)
train_ground, test_ground, index_train, index_test = separate_train_test(img_ytrain, index = i)

del(x_train)
del(y_train)
del(img_xtrain)
del(img_ytrain)

train_X, test_X, train_ground, test_ground = image_preprocess(train_X, test_X, train_ground, test_ground, size_img = size_img)

def separate_valid(train, ground, index_train, index = [1]):
    '''
    Separa os valores de Validação
    '''
    train_X = []
    train_ground = []
    train_index = []
    valid_X = []
    valid_ground = []
    valid_index = []
    for i in range(len(train)):
        if (index_train[i] not in index):
            train_X.append(train[i])
            train_ground.append(ground[i])
            train_index.append(index_train[i])
        else:
            valid_X.append(train[i])
            valid_ground.append(ground[i])
            valid_index.append(index_train[i])
    
    return np.array(train_X), np.array(valid_X), np.array(train_ground), np.array(valid_ground), train_index, valid_index

train_X, valid_X, train_ground, valid_ground, index_train, index_valid = separate_valid(train_X, train_ground, index_train = index_train, index = [2, 3])

'''
for i in range(len(x_train)):
    
    print("Rodando pela %i vez"%(i))

    train_X, valid_X, index_train_x, index_test_x = separate_train_test(img_xtrain, index = i)
    train_ground, valid_ground, index_train_y, index_test_y = separate_train_test(img_ytrain, index = i)


    train_X, valid_X, train_ground, valid_ground = image_preprocess(train_X, valid_X, train_ground, valid_ground)

    #save_nifti(predict_vol, "imagem_%i"%(i))
    #save_nifti(predict_vol, "imagem_%i"%(i))

    index_train.append(index_train_x)
    index_test.append(index_test_x)

np.savetxt('Index Train.txt', index_train)
np.savetxt('Index Test.txt', index_test)
np.savetxt('.txt', index)

'''

#Imagens tem que ser como Subj_21slice_22.png

def image_build(images, directory,img_index, size_img = 256):
    '''
    Função para realizar a criação das imagens para uma pasta no formato:
        Subj_%indexslice_%index.png
    Recebe como argumentos imagens, o diretório da imagem, o multiplicador em
    altura e largura
    '''
    indice = 0
    old_index = img_index[0]
    Subj = 0

    for j in range(len(images)):
        print("imprimindo imagem %i"%(j))
        if (img_index[j] != old_index):
            old_index = img_index[j]
            indice = 0
            Subj += 1
        
        curr_img = np.reshape(images[j], (size_img,size_img))
        #f_name = directory + "Subj_%islice_%i"%(img_index[j]+1, indice+1)
        f_name = directory + "Subj_%islice_%i"%(Subj, indice+1)
        imageio.imwrite(f_name + '.png', curr_img)
        
        indice += 1

np.savetxt('Indexes/Index Train.txt', index_train)
np.savetxt('Indexes/Index Test.txt', index_test)
np.savetxt('Indexes/Index Valid.txt', index_valid)

image_build(train_X, "image_data/train/Img/", index_train)
image_build(test_X, "image_data/test/Img/", index_test)
image_build(valid_X, "image_data/val/Img/", index_valid)

image_build(train_ground, "image_data/train/GT/", index_train)
image_build(test_ground, "image_data/test/GT/", index_test)
image_build(valid_ground, "image_data/val/GT/", index_valid)

volume = make_vol(valid_ground, index = index_valid)
save_nifti(volume, path = "image_data/GT_Nifti/Val_1", name = "Subj_0", index = 0)
save_nifti(volume, path = "image_data/GT_Nifti/Val_1", name = "Subj_1", index = 1)


