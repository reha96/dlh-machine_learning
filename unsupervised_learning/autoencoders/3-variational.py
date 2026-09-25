#!/usr/bin/env python3
# (Based on 2-convolutional.py)
"""Create a variational autoencoder model."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims: An integer containing the dimensions of the model input.
        hidden_layers: A list containing the number of nodes for each hidden
            layer in the encoder, respectively. The hidden layers should be
            reversed for the decoder.
        latent_dims: An integer containing the dimensions of the latent space
            representation.

    Returns:
        encoder, decoder, auto: The encoder model, decoder model, and full
            autoencoder model. The encoder should output the latent
            representation, the mean, and the log variance, respectively.

    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss. All layers should use a relu activation except
    for the mean and log variance layers in the encoder, which should use None,
    and the last layer in the decoder, which should use sigmoid.
    """
    pass
