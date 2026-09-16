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
    X = K.Input(shape=(224, 224, 3))

    # X = K.layers.MaxPooling2D(3, strides=2, padding='same')(X)
    # X = K.layers.AveragePooling2D(7, strides=1)(X)
    # out = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(X)
    # model = K.Model(inputs=inp, outputs=out)
