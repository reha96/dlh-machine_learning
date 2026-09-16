#!/usr/bin/env python3
"""Builds a projection block as described in Deep Residual Learning."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block.

    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization with seed zero.

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3, F12, respectively:
            F11 is the number of filters in the first 1x1 convolution.
            F3 is the number of filters in the 3x3 convolution.
            F12 is the number of filters in the second 1x1 convolution
                as well as the 1x1 convolution in the shortcut connection.
        s: stride of the first convolution in both the main path and
            the shortcut connection.

    Returns:
        The activated output of the projection block.
    """
    # same as previous task:
    # extract relevant dims
    f11, f3, f12 = filters

    # 3 blocks of 3 conv layers
    # step 1: create conv layer with kernel 1x1
    kernel_init = K.initializers.HeNormal(seed=0)
    X = K.layers.Conv2D(filters=f11, kernel_size=1, padding='same',
                        strides=s,  # stride param
                        kernel_initializer=kernel_init)(A_prev)

    # apply batch normalization and ReLu activation
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # step 2: create conv layer with kernel 3x3
    X = K.layers.Conv2D(filters=f3, kernel_size=3, padding='same',
                        kernel_initializer=kernel_init)(X)

    # apply batch normalization and ReLu activation
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # step 3: create conv layer with kernel 1x1
    X = K.layers.Conv2D(filters=f12, kernel_size=1, padding='same',
                        kernel_initializer=kernel_init)(X)

    # apply batch normalization but no ReLu before adding
    X = K.layers.BatchNormalization(axis=3)(X)

    # new block, parallel to the first one:
    X_new = K.layers.Conv2D(filters=f12, kernel_size=1, padding='same',
                            strides=s,  # stride param
                            kernel_initializer=kernel_init)(A_prev)

    # apply batch normalization but no ReLu before adding
    X_new = K.layers.BatchNormalization(axis=3)(X_new)

    # take two tensors, and add them element-wise together
    X = K.layers.Add()([X, X_new])
    # then apply ReLu
    X = K.layers.Activation('relu')(X)

    return X
