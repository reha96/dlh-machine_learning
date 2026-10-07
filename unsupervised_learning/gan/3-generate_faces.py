#!/usr/bin/env python3
# (Based on 2-wgan_gp.py)
"""Convolutional generator and discriminator for faces (GANs task 3)."""

from tensorflow import keras


def convolutional_GenDiscr():
    """Build a convolutional generator and discriminator.

    For the generator, the input data will have shape (16).
    For the discriminator, the input data will have shape (16,16,1).
    Use a hyperbolic tangent activation function "tanh" (even in Dense
    layers). In every Conv2D, the padding="same".

    Returns:
        The concatenated output of the generator and discriminator:
        the keras model gen followed by the keras model discr.
    """

    def get_generator():
        """Build the generator model; input shape (16)."""
        inputs = keras.Input(shape=(16,))
        # project the latent vector on a small feature map
        hidden = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(hidden)
        # grow the feature map with upsampling conv blocks
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)
        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """Build the discriminator model; input shape (16, 16, 1)."""
        inputs = keras.Input(shape=(16, 16, 1))
        # shrink the picture with conv and max pooling blocks
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        # classify the flattened features
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)
        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
