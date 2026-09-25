#!/usr/bin/env python3
# (Based on 0-vanilla.py)
"""Create a sparse autoencoder model."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Create a sparse autoencoder.

    Args:
        input_dims: An integer containing the dimensions of the model input.
        hidden_layers: A list containing the number of nodes for each hidden
            layer in the encoder, respectively. The hidden layers should be
            reversed for the decoder.
        latent_dims: An integer containing the dimensions of the latent space
            representation.
        lambtha: The regularization parameter used for L1 regularization on
            the encoded output.

    Returns:
        encoder, decoder, auto: The encoder model, decoder model, and sparse
            autoencoder model.

    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss. All layers should use a relu activation except
    for the last layer in the decoder, which should use sigmoid.
    """
    pass
