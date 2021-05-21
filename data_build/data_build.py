
import glob
from PIL import Image
from utils import load_images, load_images_return_shape, load_images_return_shape_contr_stre, resize_one_img
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

stages_1 = ['S28', 'S44'] # Nao precisa exagerar a normalização
stages_2 = ['S34', 'S37'] # Precisa exagerar a normalização, imagens muito escuras
stages_3 = ['S28', 'S44', 'S34', 'S37'] # Binarizacao

ORIGINAL_SIZE = 850
NEW_SIZE = 320

# Var aux ara a transposicao do S44
stack_aux = np.zeros((320,320, 1))

#################################################################################
# Etapa 1. Normalizacao, contraste 

# Estagios S28 e S44 - normais

# contador para as imagens todas
cont = 0

for stage in stages_1: # 28 e 44

    print(stage)

    stack_glob = sorted(glob.glob('./dados_girino/A0_Filtered_images/' + stage + '/*'))
    stack_norm, img_shape = load_images_return_shape(stack_glob, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE) # Em memória já está no padrão da Unetp´
    
    #print (stack_norm.shape)
    #print (stack_aux.shape)

    #io.imshow(stack_aux)
    #plt.show()

    if (stage == 'S28'):

        for i in range(len(stack_norm)):
            io.imsave('./dados_girino/A1_Norm_images/norm_image_%s.png'%str("%03d" % cont), resize_one_img(stack_norm[i], img_shape[1], img_shape[0])) # S28
            cont += 1
    
    if (stage == 'S44'):

        img_shape = img_shape[::-1] # Ok, testada
        
        for i in range(len(stack_norm)):
            
            # Tratamento especial de transposicao para S44
            
            # Redimensiona e transpõe
            stack_aux = stack_norm[i]
            stack_aux = stack_aux.reshape(NEW_SIZE,NEW_SIZE)  
            stack_aux = np.transpose(stack_aux)

            #io.imshow(stack_aux)
            #plt.show()
            
            # Volta a img transposta para o padrão keras
            stack_aux = stack_aux.reshape(NEW_SIZE,NEW_SIZE, 1)
            stack_norm[i] = stack_aux
            
            io.imsave('./dados_girino/A1_Norm_images/norm_image_%s.png'%str("%03d" % cont), resize_one_img(stack_norm[i], img_shape[1], img_shape[0])) 
            cont += 1
              

# Estagios S34 e S37 - escuros        

for stage in stages_2: # 34 e 37
    
    print(stage)

    stack_glob = sorted(glob.glob('./dados_girino/A0_Filtered_images/' + stage + '/*'))
    stack_norm, img_shape = load_images_return_shape_contr_stre(stack_glob, size_img = ORIGINAL_SIZE, new_size = NEW_SIZE) # Em memória já está no padrão da Unetp´
  
    for i in range(len(stack_norm)):
        io.imsave('./dados_girino/A1_Norm_images/norm_image_%s.png'%str("%03d" % cont), resize_one_img(stack_norm[i], img_shape[1], img_shape[0])) 
        cont += 1
     
# Fim da 1a. etapa - normalizacao, contraste 


#################################################################################
# 2a. etapa - Binarizacao - 2021.05.01

output = './dados_girino/A2_GT_images/'
cont = 0

for stage in stages_3: 
    
    print(stage)
    stack_glob = sorted(glob.glob('./dados_girino/A0_Avizo_images/Avizo_' + stage + '/*'))

    for i in range(len(stack_glob)):
        temp = Image.open(stack_glob[i])
        print(type(temp))
        temp = temp.convert('L')
        temp = temp.point(lambda x: 255 if x<196 else 0, '1')

        if (stage == 'S44'):
            temp = temp.transpose(method = Image.TRANSPOSE)

        temp.save(output + "GT_bin_%s.tif"%str("%03d" % cont), "TIFF")

        cont += 1


# Fim da 2a. etapa - Binarizacao - 2021.05.01
#################################################################################

