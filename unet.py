# -*- coding: utf-8 -*-

from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from keras.initializers import he_normal
from keras.optimizers import Adam, RMSprop
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f))

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def unet_completa(size_img, _seed = 1):
    
    input_size = (size_img,size_img,1)
    _initializer = he_normal(seed = _seed)

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(inputs)
    #conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv1)
    #conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    	
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(pool1)
    #conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv2)
    #conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    	
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(pool2)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv3)
    #conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    	
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(pool3)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv4)
    #conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(pool4)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv5)
    #conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(merge6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv6)
    #conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(merge7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv7)
    #conv7 = BatchNormalization()(conv7)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(merge8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = _initializer)(conv8)
    #conv8 = BatchNormalization()(conv8)
    
    up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = _initializer)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = _initializer)(merge9)
    #conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = _initializer)(conv9)
    #conv9 = BatchNormalization()(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics=[dice_coef])
    
    return model
