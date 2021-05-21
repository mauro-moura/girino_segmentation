# -*- coding: utf-8 -*-

from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f))

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def unet_completa(size_img, SEED = 1):
    CONCAT_AXIS = -1
    INITIALIZER = he_normal(seed = SEED)

    input_size = (size_img, size_img, 1)

    inputs = Input(shape = (input_size))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    	
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    	
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    	
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = CONCAT_AXIS)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = CONCAT_AXIS)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv7)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = CONCAT_AXIS)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv8)
    
    up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = CONCAT_AXIS)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = INITIALIZER)(merge9)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = INITIALIZER)(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics=[dice_coef])
    
    return model

def unet_mini(size_img, SEED = 1):
    MERGE_AXIS = -1
    INITIALIZER = he_normal(seed = SEED)
    input_size = (size_img, size_img, 1)

    inputs = Input(shape = (input_size))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=MERGE_AXIS)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=MERGE_AXIS)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',
                  kernel_initializer = INITIALIZER)(conv5)

    o = Conv2D(1, (1, 1), activation = 'sigmoid', padding='same')(conv5)
    model = Model(input = inputs, output = o)
    model.compile(optimizer = Adam(lr = 1e-5),
                  loss = dice_coef_loss, metrics=[dice_coef])
    return model
