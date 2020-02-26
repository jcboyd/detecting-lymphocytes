from __future__ import print_function
from __future__ import division

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, UpSampling2D
from keras.layers import Conv2DTranspose, concatenate, Concatenate
from keras.layers import Flatten, Reshape, Dense, Dropout, BatchNormalization
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
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

    generator.add(Dense(256, input_dim=noise_dim,
                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(Dense(784,
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

    x = concatenate([z_input, y_input])

    x = Dense(256, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Dense(784, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Dense(output_dim, activation='tanh',
              kernel_initializer='glorot_uniform')(x)
    x = Reshape(img_shape)(x)

    generator = Model(inputs=[z_input, y_input], outputs=x)

    x_input = Input(img_shape)
    y_input = Input((nb_classes,))

    x = concatenate([Flatten()(x_input), y_input])

    x = Dense(512, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)

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


def patch_gan(image_shape, label_shape):

    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    images = Input(shape=image_shape)
    labels = Input(shape=label_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([images, labels])

    d1 = d_layer(combined_imgs, 64, bn=False)
    d2 = d_layer(d1, 128)
    d3 = d_layer(d2, 256)
    d4 = d_layer(d3, 512)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([images, labels], validity)


def patch_gan_cycle(img_shape):

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=img_shape)

    d1 = d_layer(img, 64, normalization=False)
    d2 = d_layer(d1, 128)
    d3 = d_layer(d2, 256)
    d4 = d_layer(d3, 512)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)


def fnet(img_shape, num_filters, activation):

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, num_filters, bn=False)
    d2 = conv2d(d1, num_filters * 2)
    d3 = conv2d(d2, num_filters * 4)
    d4 = conv2d(d3, num_filters * 8)
    d5 = conv2d(d4, num_filters * 8)
    d6 = conv2d(d5, num_filters * 8)
    d7 = conv2d(d6, num_filters * 8)

    # Upsampling
    u1 = deconv2d(d7, d6, num_filters * 8)
    u2 = deconv2d(u1, d5, num_filters * 8)
    u3 = deconv2d(u2, d4, num_filters * 8)
    u4 = deconv2d(u3, d3, num_filters * 4)
    u5 = deconv2d(u4, d2, num_filters * 2)
    u6 = deconv2d(u5, d1, num_filters)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation=activation)(u7)

    return Model(d0, output_img)


def fnet_cycle(img_shape, num_filters):

    def conv2d(layer_input, filters, strides=2, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    input_img = Input(shape=img_shape)

    in_conv_1 = conv2d(input_img, num_filters, strides=1)
    in_conv_2 = conv2d(in_conv_1, num_filters, strides=1)

    # Downsampling
    d1 = conv2d(in_conv_2, num_filters)
    d2 = conv2d(d1, num_filters * 2)
    d3 = conv2d(d2, num_filters * 4)
    d4 = conv2d(d3, num_filters * 8)
    d5 = conv2d(d4, num_filters * 8)
    d6 = conv2d(d5, num_filters * 8)

    # Upsampling
    u1 = deconv2d(d6, d5, num_filters * 8)
    u2 = deconv2d(u1, d4, num_filters * 8)
    u3 = deconv2d(u2, d3, num_filters * 4)
    u4 = deconv2d(u3, d2, num_filters * 2)
    u5 = deconv2d(u4, d1, num_filters)

    u6 = UpSampling2D(size=2)(u5)

    out_conv_1 = conv2d(u6, num_filters, strides=1)
    out_conv_2 = conv2d(out_conv_1, num_filters, strides=1)

    output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(out_conv_2)

    return Model(input_img, output_img)
