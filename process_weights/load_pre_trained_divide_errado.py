
import glob

import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import backend as K
K.clear_session()

from utils import resize_img
from utils import create_folder, resize_one_img, load_images_array, load_images_array_return_shape

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true) # GT
    y_pred_f = K.flatten(y_pred) # Predicted
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f))

def cut_img(img, offset = 0):
    img = np.reshape(img, (img.shape[0],img.shape[1]))
    height, width = img.shape
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff - offset]
    s2 = img[:, width_cutoff + offset:]
    return s1, s2

def cut_img_list(data, offset = 0):
    s1_list = []
    s2_list = []
    for i in range(len(data)):
        s1, s2 = cut_img(data[i], offset)
        s1_list.append(s1)
        s2_list.append(s2)
    
    return s1_list, s2_list

def reshape_parts(conc):
    conc = np.asarray(conc)
    #print(conc.shape)
    conc = np.reshape(conc, (conc.shape[0],conc.shape[1], conc.shape[2], 1))
    return conc

def rebuild_parts(img, img_ld, img_lu, total_imgs, size_img = 256, offset = 48):
    ld_s1 = np.zeros((img_ld, size_img, int(size_img/2 - offset)))
    lu_s1 = np.zeros((total_imgs - img_lu, size_img, int(size_img/2 - offset)))

    new_img = np.concatenate((ld_s1, img, lu_s1), axis = 0)

    return new_img

ORIGINAL_SIZE = 850 #Antigo Size Img
NEW_SIZE = 256 #Tamanho para qual as imagens serão convertidas, deixe igual ao original se não for alterar

working_folder = './TM40_46prod_Mauro_SingleEye/'
batch = [4, 8]
index = 0

print("Carregando novas imagens")
_folder = "./dados_girino/TM46"
new_imgs = sorted(glob.glob(_folder + '/Producao/*'))
X , img_shape = load_images_array_return_shape(new_imgs, ORIGINAL_SIZE, NEW_SIZE)

GT_Test = sorted(glob.glob(_folder + '/GT_Producao/*'))
Y = load_images_array(GT_Test, new_size = NEW_SIZE)

offset = 48
total_imgs = len(new_imgs)
s1, s2 = cut_img_list(X, offset)

# TM40
#s1_ld, s1_lu, s2_ld, s2_lu = 0, 276, 110, 387
# TM46
s1_ld, s1_lu, s2_ld, s2_lu = 89, 408, 0, 326

s1 = s1[s1_ld:s1_lu]
s2 = s2[s2_ld:s2_lu]

s1 = resize_img(s1, NEW_SIZE, NEW_SIZE)
s2 = resize_img(s2, NEW_SIZE, NEW_SIZE)

s1 = reshape_parts(s1)
s2 = reshape_parts(s2)

n_exec = 1
n_fold = 10

for i in range(n_fold):
    new_imgs_load = X
    GT_Test_dice = Y

    print("\n\n\nRealizando Execução: %i\n\n\n"%n_exec)
    execution_name = 'Exec_%i'%n_exec
    folder_name = working_folder + execution_name + '/'
    filename = ['girino_4_100_%s'%execution_name.lower()]

    model = keras.models.load_model(folder_name + '%s.h5'%(filename[index]), compile=False)

    s1_pred = model.predict(s1)
    s1_pred = resize_img(s1_pred, int(NEW_SIZE/2 - offset), NEW_SIZE)
    s1_pred = rebuild_parts(s1_pred, s1_ld, s1_lu, total_imgs)
    print(s1_pred.shape)

    middle = np.zeros((total_imgs, NEW_SIZE, 2 * offset))

    s2_pred = model.predict(s2)
    s2_pred = resize_img(s2_pred, int(NEW_SIZE/2 - offset), NEW_SIZE)
    s2_pred = rebuild_parts(s2_pred, s2_ld, s2_lu, total_imgs)


    new_predicao = np.concatenate((s1_pred, middle, s2_pred), axis = -1)
    print(new_predicao.shape)
    new_predicao = new_predicao > 0.5
    new_predicao = np.float64(new_predicao)

    #fig, ax = plt.subplots()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    plt.title('Imagem')
    ax1.imshow(X[10])
    ax2 = fig.add_subplot(2, 2, 2)
    plt.title('Predict')
    ax2.imshow(new_predicao[10])
    ax3 = fig.add_subplot(2, 2, 3)
    plt.title('GT')
    ax3.imshow(GT_Test_dice[10])
    plt.show()

    '''
    plt.imshow(s1[10])
    plt.show()
    plt.imshow(new_predicao[10])
    plt.show()
    plt.imshow(GT_Test_dice[10])
    plt.show()
    '''

    print("Predizendo " + str(len(new_predicao)) + " Imagens")
    create_folder(folder_name + 'outputs_prod')
    for i in range(len(new_predicao)):
        io.imsave(folder_name + 'outputs_prod/predicao_%s_%s.png'%(str(GT_Test[i][-7:-4]), str(batch[index])), resize_one_img(new_predicao[i], img_shape[1], img_shape[0]))
        #io.imsave(folder_name + 'outputs_prod/predicao_%s_%s.png'%((i), str(batch[index])), resize_one_img(new_predicao[i], img_shape[1], img_shape[0]))

    print("Calculando o dice de produção")
    dice_metric = []
    sess = tf.InteractiveSession()
    for i in range(len(new_predicao)):
        dice_metric.append(dice_coef(new_predicao[i], GT_Test_dice[i]).eval())
        print("Dice número", i, " = ", dice_metric[i])
    sess.close()

    print('Salvando valores de Dice...\nMédia dos Dices: ' + str(np.mean(dice_metric)))
    with open(folder_name + 'outputs_prod/dice_metric_production_%s.txt'%(str(batch[index])), 'w') as file:
        file.write(str(dice_metric))
    
    K.clear_session()
    n_exec += 1
