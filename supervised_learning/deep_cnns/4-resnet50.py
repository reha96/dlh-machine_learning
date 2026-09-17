#!/usr/bin/env python3
"""Builds the ResNet-50 architecture as described in Deep Residual Learning."""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture.

    Returns:
        The Keras model.
    """

    # building 50-layer ResNet
    # start with input
    A_prev = K.Input(shape=(224, 224, 3))

    # build the stem: conv1 from Table 1
    # init as before
    kernel_init = K.initializers.HeNormal(seed=0)
    # from Table 1: 7x7 dim, 64 times, stride 2
    X = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                        padding='same', kernel_initializer=kernel_init)(A_prev)

    # apply batch normalization and ReLu activation
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # conv2
    # apply pooling
    # from Table 1: 3x3 max pool, stride 2
    X = K.layers.MaxPooling2D(3, strides=2, padding='same')(X)

    # projection block with nb filters=64,64,256 X1, stride=1
    filter_cv2 = (64, 64, 256)
    X = projection_block(X, filter_cv2, s=1)
    # then add identity block X2
    X = identity_block(X, filter_cv2)
    X = identity_block(X, filter_cv2)

    # conv3
    # projection block with nb filters=128,128,512 X1, stride=2
    filter_cv3 = (128, 128, 512)
    X = projection_block(X, filter_cv3, s=2)
    # then add identity block X3
    X = identity_block(X, filter_cv3)
    X = identity_block(X, filter_cv3)
    X = identity_block(X, filter_cv3)

    # conv4
    # projection block with nb filters=256,256,1024 X1, stride=2
    filter_cv4 = (256, 256, 1024)
    X = projection_block(X, filter_cv4, s=2)
    # then add identity block X5
    X = identity_block(X, filter_cv4)
    X = identity_block(X, filter_cv4)
    X = identity_block(X, filter_cv4)
    X = identity_block(X, filter_cv4)
    X = identity_block(X, filter_cv4)

    # conv5
    # projection block with nb filters=512,512,2048 X1, stride=2
    filter_cv5 = (512, 512, 2048)
    X = projection_block(X, filter_cv5, s=2)
    # then add identity block X2
    X = identity_block(X, filter_cv5)
    X = identity_block(X, filter_cv5)

    # average pool from 7x7 to 1x1
    X = K.layers.AveragePooling2D(7, strides=1)(X)
    # 1000 class softmax output
    out = K.layers.Dense(1000, activation='softmax',
                         kernel_initializer=kernel_init
                         )(X)
    model = K.Model(inputs=A_prev, outputs=out)

    return model
