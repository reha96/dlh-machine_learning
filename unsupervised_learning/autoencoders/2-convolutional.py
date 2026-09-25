#!/usr/bin/env python3
# (Based on 1-sparse.py)
"""Create a convolutional autoencoder model."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder.

    Args:
        input_dims: A tuple of integers containing the dimensions of the model
            input.
        filters: A list containing the number of filters for each convolutional
            layer in the encoder, respectively. The filters should be reversed
            for the decoder.
        latent_dims: A tuple of integers containing the dimensions of the
            latent space representation.

    Returns:
        encoder, decoder, auto: The encoder model, decoder model, and full
            autoencoder model.

    Each encoder convolution should use a (3, 3) kernel, same padding, and relu
    activation, followed by max pooling of size (2, 2). Each decoder
    convolution except the last two should use a (3, 3) filter, same padding,
    and relu activation, followed by upsampling of size (2, 2). The second to
    last convolution should use valid padding. The last convolution should use
    the input channel count and sigmoid activation, with no upsampling. The
    autoencoder should be compiled using adam optimization and binary
    cross-entropy loss.
    """
    pass
