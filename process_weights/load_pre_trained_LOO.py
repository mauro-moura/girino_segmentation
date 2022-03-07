
import glob

import tensorflow as tf
import numpy as np
from skimage import io

from tensorflow import keras
from keras import backend as K
K.clear_session()

from utils import create_folder, resize_one_img, load_images_array, load_images_array_return_shape

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true) # GT
    y_pred_f = K.flatten(y_pred) # Predicted
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f))

ORIGINAL_SIZE = 850 #Antigo Size Img
NEW_SIZE = 320 #Tamanho para qual as imagens serão convertidas, deixe igual ao original se não for alterar

working_folder = './Teste_2/'
batch = [4, 8]
index = 0

print("Carregando novas imagens")
_folder = "./dados_girino/Teste_2"
new_imgs = sorted(glob.glob(_folder + '/Producao/*'))
X , img_shape = load_images_array_return_shape(new_imgs, ORIGINAL_SIZE, NEW_SIZE)

GT_Test = sorted(glob.glob(_folder + '/GT_Producao/*'))
Y = load_images_array(GT_Test, new_size = NEW_SIZE)

n_exec = 1
n_fold = 5

# ********** SOMENTE PARA O CORAÇÃO **********
_tam = [67, 70, 73, 76, 78, 58, 67, 65, 61, 62, 67, 73, 69, 70, 57, 53, 69, 72, 59, 68]
_tam_now = 0
def leave_um_fora(list_orig, trim_a, trim_b):
    list1 = list_orig[trim_a:trim_b]
    #list2 = list_orig[:trim_b]
    return list1 #np.concatenate((list1, list2), axis=0)
Y = Y > 0
# ********** SOMENTE PARA O CORAÇÃO **********

for i in range(n_fold):
    # ********** SOMENTE PARA O CORAÇÃO **********
    new_imgs_load = leave_um_fora(X, _tam_now, _tam_now + _tam[n_exec - 1])
    GT_Test_dice = leave_um_fora(Y, _tam_now, _tam_now + _tam[n_exec - 1])
    GT_Test_dice = np.float64(GT_Test_dice)

    _tam_now += _tam[n_exec - 1]
    # ********** SOMENTE PARA O CORAÇÃO **********

    print("\n\n\nRealizando Execução: %i\n\n\n"%n_exec)
    execution_name = 'Exec_%i'%n_exec
    folder_name = working_folder + execution_name + '/'
    filename = ['girino_4_100_%s'%execution_name.lower()]

    model = keras.models.load_model(folder_name + '%s.h5'%(filename[index]), compile=False)

    new_predicao = model.predict(new_imgs_load)
    new_predicao = new_predicao > 0.5
    new_predicao = np.float64(new_predicao)

    print("Predizendo " + str(len(new_predicao)) + " Imagens")
    create_folder(folder_name + 'outputs_prod')
    for i in range(len(new_predicao)):
        io.imsave(folder_name + 'outputs_prod/predicao_%s_%s.png'%(str(GT_Test[i][-7:-4]), str(batch[index])), resize_one_img(new_predicao[i], img_shape[1], img_shape[0]))

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
