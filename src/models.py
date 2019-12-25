from __future__ import print_function
from __future__ import division

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, UpSampling2D
from keras.layers import Conv2DTranspose, concatenate
from keras.layers import Flatten, Reshape, Dense, Dropout, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import SGD
import keras.backend as K


def MultiFCN(input_shape=(None, None, 1), nb_classes=2, l=3e-5):

    loss_mask = Input(shape=(1, 1, 2))

    input_img = Input(shape=input_shape, name='input')

    x = Conv2D(16, kernel_size=(3, 3),
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, kernel_size=(3, 3),
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3),
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=(3, 3),
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3),
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=(1, 1),
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='cnn_code')(x)

    x_output_obj = Conv2D(2, kernel_size=(1, 1), name='logits_obj',
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x_output_obj = Activation('softmax', name='o')(x_output_obj)

    x_output_cls = Conv2D(nb_classes, kernel_size=(1, 1), name='logits_cls',
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x_output_cls = Activation('softmax', name='c')(x_output_cls)

    x_output_bbs = Conv2D(nb_classes, kernel_size=(1, 1), name='logits_bbs',
               kernel_regularizer=l2(l),
               kernel_initializer='glorot_uniform')(x)
    x_output_bbs = Activation('linear', name='b')(x_output_bbs)

    model_tr = Model(inputs=[input_img, loss_mask],
                     outputs=[x_output_obj, x_output_cls, x_output_bbs])

    model = Model(inputs=input_img,
                  outputs=[x_output_obj, x_output_cls, x_output_bbs])


    def weighted_mse(y_true, y_pred):

        return K.mean(loss_mask * K.square(y_true - y_pred), axis=-1)


    def weighted_log_loss(y_true, y_pred):

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)

        return -K.mean(loss_mask * y_true * K.log(y_pred), axis=-1)


    opt = SGD(lr=5e-3, momentum=0.9, nesterov=True, decay=1e-5)
    losses = ['categorical_crossentropy', weighted_log_loss, weighted_mse]
    model_tr.compile(loss=losses, optimizer=opt)
    
    return model_tr, model


def SoftmaxRegression(input_shape=(2048,), nb_classes=3):

    x_input = Input(shape=input_shape, name='input')

    x = Dense(nb_classes, activation='softmax')(x_input)
    model = Model(inputs=x_input, outputs=x)

    return model


def GAN(img_shape, noise_dim):

    output_dim = np.prod(img_shape)

    # Generator
    generator = Sequential()

    generator.add(Dense(512, input_dim=noise_dim,
                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(Dense(1024,
                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(Dense(output_dim, activation='tanh',
                        kernel_initializer='glorot_uniform'))
    generator.add(Reshape(img_shape))

    # Discriminator
    discriminator = Sequential()

    discriminator.add(Flatten())

    discriminator.add(Dense(512,
                      kernel_initializer='glorot_uniform'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256,
                      kernel_initializer='glorot_uniform'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid',
                      kernel_initializer='glorot_uniform'))

    return generator, discriminator
 

def CGAN(img_shape, noise_dim, nb_classes):

    output_dim = np.prod(img_shape)

    z_input = Input((noise_dim,))
    y_input = Input((nb_classes,))

    z = Dense(128, input_dim=noise_dim,
              kernel_initializer='glorot_uniform')(z_input)
    z = BatchNormalization(momentum=0.9)(z)
    z = Activation('relu')(z)

    y = Dense(128, input_dim=nb_classes,
          kernel_initializer='glorot_uniform')(y_input)
    y = BatchNormalization(momentum=0.9)(y)
    y = Activation('relu')(y)

    x = concatenate([z, y])

    x = Dense(512, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Dense(output_dim, activation='tanh',
              kernel_initializer='glorot_uniform')(x)
    x = Reshape(img_shape)(x)

    generator = Model(inputs=[z_input, y_input], outputs=x)

    x_input = Input(img_shape)
    y_input = Input((nb_classes,))

    x = Flatten()(x_input)
    x = Dense(256, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)

    y = Dense(256, kernel_initializer='glorot_uniform')(y_input)
    y = LeakyReLU(0.2)(y)

    x = concatenate([x, y])

    x = Dense(256, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    discriminator = Model(inputs=[x_input, y_input], outputs=x)

    return generator, discriminator


def DCGAN(img_shape, noise_dim):

    # Generator
    generator = Sequential()

    s = img_shape[0] // 4
    nb_channels = img_shape[-1]

    generator.add(Dense(128 * s * s, input_dim=noise_dim,
                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.9))  # changing dist. merits lower momentum
    generator.add(Activation('relu'))

    generator.add(Reshape((s, s, 128)))

    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(64, kernel_size=(5, 5), padding='same',
                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(nb_channels, kernel_size=(5, 5), padding='same', activation='tanh',
                  kernel_initializer='glorot_uniform'))

    # Discriminator
    discriminator = Sequential()

    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
                      padding='same', input_shape=img_shape,
                      kernel_initializer='glorot_uniform'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2),
                      padding='same', kernel_initializer='glorot_uniform'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid',
                      kernel_initializer='glorot_uniform'))

    return generator, discriminator


def CDCGAN(img_shape, noise_dim, nb_classes):

    output_dim = np.prod(img_shape)
    s = img_shape[0] // 4
    nb_channels = img_shape[-1]

    z_input = Input((noise_dim,))
    y_input = Input((nb_classes,))

    # z = Dense(64 * s * s, input_dim=noise_dim,
    #           kernel_initializer='glorot_uniform')(z_input)
    # z = BatchNormalization(momentum=0.9)(z)
    # z = Activation('relu')(z)

    # y = Dense(64 * s * s, input_dim=nb_classes,
    #           kernel_initializer='glorot_uniform')(y_input)
    # y = BatchNormalization(momentum=0.9)(y)
    # y = Activation('relu')(y)

    # x = concatenate([z, y])

    x = concatenate([z_input, y_input])

    x = Dense(128 * s * s,
              kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Reshape((s, s, 128))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(5, 5), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(nb_channels, kernel_size=(5, 5), padding='same', activation='tanh',
               kernel_initializer='glorot_uniform')(x)

    generator = Model(inputs=[z_input, y_input], outputs=x)

    x_input = Input(img_shape)
    y_input = Input((nb_classes,))

    x = Conv2D(16, kernel_size=(5, 5), strides=(2, 2),
               padding='same', input_shape=img_shape,
               kernel_initializer='glorot_uniform')(x_input)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(16, kernel_size=(5, 5), strides=(2, 2),
               padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    x = Dense(128, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)

    y = Dense(128, kernel_initializer='glorot_uniform')(y_input)
    y = LeakyReLU(0.2)(y)

    x = concatenate([x, y])

    x = Dense(128, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(1, activation='sigmoid',
              kernel_initializer='glorot_uniform')(x)

    discriminator = Model(inputs=[x_input, y_input], outputs=x)

    return generator, discriminator
