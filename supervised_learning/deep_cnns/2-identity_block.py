#!/usr/bin/env python3
"""Builds an identity block as described in Deep Residual Learning."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block.

    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization with seed zero.

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3, F12, respectively:
            F11 is the number of filters in the first 1x1 convolution.
            F3 is the number of filters in the 3x3 convolution.
            F12 is the number of filters in the second 1x1 convolution.

    Returns:
        The activated output of the identity block.
    """
    # extract relevant dims
    f11, f3, f12 = filters

    # 3 blocks of 3 conv layers
    # step 1: create conv layer with kernel 1x1
    kernel_init = K.initializers.HeNormal(seed=0)
    X = K.layers.Conv2D(filters=f11, kernel_size=1, padding='same',
                        kernel_initializer=kernel_init)(A_prev)

    # apply batch normalization and ReLu activation
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # step 2: create conv layer with kernel 3x3
    X = K.layers.Conv2D(filters=f3, kernel_size=3, padding='same',
                        kernel_initializer=K.initializers.HeNormal(seed=0))(X)

    # apply batch normalization and ReLu activation
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # step 3: create conv layer with kernel 1x1
    X = K.layers.Conv2D(filters=f12, kernel_size=1, padding='same',
                        kernel_initializer=K.initializers.HeNormal(seed=0))(X)

    # apply batch normalization but no ReLu before adding
    X = K.layers.BatchNormalization(axis=3)(X)

    # take two tensors, and add them element-wise together
    K.Add()([X, A_prev])
    # then apply ReLu
    X = K.layers.Activation('relu')(X)

    return X
