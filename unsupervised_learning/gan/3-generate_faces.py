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
        # Dense + Reshape, then upsampling conv blocks with tanh
        # activation, padding="same" and BatchNormalization
        pass

    def get_discriminator():
        """Build the discriminator model; input shape (16, 16, 1)."""
        # conv blocks with tanh activation and padding="same",
        # then Flatten and a Dense output layer
        pass

    return get_generator(), get_discriminator()
