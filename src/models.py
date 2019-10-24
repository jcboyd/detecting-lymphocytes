from __future__ import print_function
from __future__ import division

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation
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
