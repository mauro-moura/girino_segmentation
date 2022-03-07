from keras.models import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras import backend as K

def dice_coef(y_true, y_pred, smooth = 0.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def unet_hedden(size_img, SEED = 1):
    CONCAT_AXIS = -1
    INITIALIZER = he_normal(seed = SEED)
    input_size = (size_img,size_img,1)

    inputs = Input(input_size)

    # DownConv
    conv1 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(inputs)
    conv1 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool1)
    conv2 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv2)
    drop2 = Dropout(0.5, seed = SEED)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool2)
    conv3 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv3)
    drop3 = Dropout(0.5, seed = SEED)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool3)
    conv4 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv4)
    drop4 = Dropout(0.5, seed = SEED)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool4)
    conv5 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv5)
    drop5 = Dropout(0.5, seed = SEED)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

    conv6 = Conv2D(768, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool5)
    conv6 = Conv2D(768, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv6)
    drop6 = Dropout(0.5, seed = SEED)(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(drop6)

    # Camada do Meio
    conv7 = Conv2D(1536, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(pool6)
    conv7 = Conv2D(1536, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv7)
    drop7 = Dropout(0.5, seed = SEED)(conv7)

    # UpConv
    #up8 = Conv2DTranspose(768, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(drop7))
    up8 = Conv2D(768, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(drop7))
    dropup8 = Dropout(0.5, seed = SEED)(up8)
    merge8 = concatenate([drop6,dropup8], axis = CONCAT_AXIS)
    conv8 = Conv2D(768, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge8)
    conv8 = Conv2D(768, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv8)

    #up9 = Conv2DTranspose(384, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2D(384, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv8))
    dropup9 = Dropout(0.5, seed = SEED)(up9)
    merge9 = concatenate([drop5,dropup9], axis = CONCAT_AXIS)
    conv9 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge9)
    conv9 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv9)

    #up10 = Conv2DTranspose(192, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv9))
    up10 = Conv2D(192, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv9))
    dropup10 = Dropout(0.5, seed = SEED)(up10)
    merge10 = concatenate([drop4,dropup10], axis = CONCAT_AXIS)
    conv10 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge10)
    conv10 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv10)

    #up11 = Conv2DTranspose(96, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv10))
    up11 = Conv2D(96, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv10))
    dropup11 = Dropout(0.5, seed = SEED)(up11)
    merge11 = concatenate([drop3,dropup11], axis = CONCAT_AXIS)
    conv11 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge11)
    conv11 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv11)

    #up12 = Conv2DTranspose(48, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv11))
    up12 = Conv2D(48, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv11))
    dropup12 = Dropout(0.5, seed = SEED)(up12)
    merge12 = concatenate([drop2,dropup12], axis = CONCAT_AXIS)
    conv12 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge12)
    conv12 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv12)

    # Saida
    #up13 = Conv2DTranspose(24, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv12))
    up13 = Conv2D(24, 2, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(UpSampling2D(size = (2,2))(conv12))
    merge13 = concatenate([conv1,up13], axis = CONCAT_AXIS)
    conv13 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(merge13)
    conv13 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv13)
    conv13 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv13)
    conv13 = Conv2D(12, 3, activation = 'relu', padding = 'same', kernel_initializer = INITIALIZER)(conv13)

    out = Conv2D(1, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = INITIALIZER)(conv13)

    model = Model(input = inputs, output = out)

    model.compile(optimizer = Adam(lr = 0.0001), loss = dice_coef_loss, metrics=[dice_coef])
    
    return model