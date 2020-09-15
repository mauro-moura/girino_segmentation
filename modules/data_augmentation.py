
import numpy as np
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class MauroDataGenerator():
    def __init__(self, seed=42):
        self.datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=2)
        self.seed = seed
    
    def simple_load_images(self, img_list):
        img = []
        for i in range(len(img_list)):
            img.append(io.imread(img_list[i]))
        img = np.float64(img)
        img = img.reshape(-1, img.shape[-2], img.shape[-1], 1)
        return img

    def data_agumentation_2D(self, filenames_x, filenames_y, x, y, datagen,
                        seed = 42, n_img = 8, size_img = 128, save=False,
                        train_path= './Test/Aug_Train', gt_path= './Test/Aug_GT'):
        
        images = []
        masks = []
        len_input = len(x)

        for j in range(len_input):
            image = x[j].reshape((1, ) + x[j].shape)
            mask = y[j].reshape((1, ) + y[j].shape)

            i = 0
            for batch_x in (datagen.flow(image, batch_size= 1, seed=seed)):
                images.insert(j+len_input*i, batch_x.reshape(size_img, size_img, 1))
                if (save):
                    FNAME_X = train_path + str(i) + filenames_x[j][8:-4] + '.tiff'
                    batch_x = batch_x.reshape(size_img, size_img)
                    plt.imsave(FNAME_X, batch_x, cmap= 'gray', format='tiff')
                i += 1
                if i >= n_img:
                    break
            
            i = 0
            for batch_y in (datagen.flow(mask, batch_size= 1, seed=seed)):
                masks.insert(j+len_input*i, batch_y.reshape(size_img, size_img, 1))
                if (save):
                    FNAME_Y = gt_path + str(i) + filenames_y[j][5:-4] + '.tiff'
                    batch_y = batch_y.reshape(size_img, size_img)
                    plt.imsave(FNAME_Y, batch_y, cmap= 'gray', format='tiff')
                i += 1
                if i >= n_img:
                    break
        
        return images, masks
    
    def data_agumentation_3D(self, filenames_x, filenames_y, x, y, datagen,
                        seed = 42, n_img = 8, size_img = 128, save=False,
                        train_path= './Test/Aug_Train', gt_path= './Test/Aug_GT'):
        
        images = np.empty((n_img, 0)).tolist()
        masks = np.empty((n_img, 0)).tolist()

        for j in range(len(x)):
            image = x[j].reshape((1, ) + x[j].shape)
            mask = y[j].reshape((1, ) + y[j].shape)

            i = 0
            for batch_x in (datagen.flow(image, batch_size= 1, seed=seed)):
                images[i].append(batch_x.reshape(size_img, size_img, 1))
                if (save):
                    FNAME_X = train_path + str(i) + filenames_x[j][8:-4] + '.tiff'
                    batch_x = batch_x.reshape(size_img, size_img)
                    plt.imsave(FNAME_X, batch_x, cmap= 'gray', format='tiff')
                i += 1
                if i >= n_img:
                    break
            
            i = 0
            for batch_y in (datagen.flow(mask, batch_size= 1, seed=seed)):
                masks[i].append(batch_y.reshape(size_img, size_img, 1))
                if (save):
                    FNAME_Y = gt_path + str(i) + filenames_y[j][5:-4] + '.tiff'
                    batch_y = batch_y.reshape(size_img, size_img)
                    plt.imsave(FNAME_Y, batch_y, cmap= 'gray', format='tiff')
                i += 1
                if i >= n_img:
                    break
        
        return images, masks
    
    def run_all(self, filenames_x, filenames_y, x=np.empty(0), y=np.empty(0),
                    n_img=8, size_img=850, save=False,
                    train_path='./dados_girino/Aug_Train',
                    gt_path='./dados_girino/Aug_GT',
                    Aug_2D=True):
        if (x.size == 0 and y.size == 0):
            x = self.simple_load_images(filenames_x)
            y = self.simple_load_images(filenames_y)

        if (Aug_2D):
            images, masks = self.data_agumentation_2D(filenames_x, filenames_y, x, y, self.datagen, seed=self.seed,
                                 n_img=n_img, size_img=size_img, save=save, train_path=train_path,
                                 gt_path=gt_path)
        else:
            images, masks = self.data_agumentation_3D(filenames_x, filenames_y, x, y, self.datagen, seed=self.seed,
                                 n_img=n_img, size_img=size_img, save=save, train_path=train_path,
                                 gt_path=gt_path)
        
        return np.float64(images), np.float64(masks)
