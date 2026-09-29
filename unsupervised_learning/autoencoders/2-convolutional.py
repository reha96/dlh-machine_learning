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

    # encoder
    # our input
    X = keras.Input(input_dims)
    inputs_e = X
    # input filters
    for f in filters:
        # conv 3,3
        X = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                activation='relu', padding='same')(X)
        # max pool 2,2
        X = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(X)
    # output is already the bottleneck
    outputs_e = X
    # create encoder mopdel
    encoder = keras.Model(inputs_e, outputs_e)

    # decoder
    # input for decoder model
    X = keras.Input(latent_dims)
    inputs_d = X

    # reversed filters on input
    # loop until second last element
    for f in reversed(filters[1:]):
        X = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                activation='relu', padding='same')(X)
        X = keras.layers.UpSampling2D(size=(2, 2))(X)
    # second to last layer
    X = keras.layers.Conv2D(filters=filters[-1], kernel_size=(
        3, 3), activation='relu', padding='valid')(X)

    # last conv layer
    X = keras.layers.Conv2D(
        filters=input_dims[-1], kernel_size=(3, 3), activation='sigmoid', padding='same')(X)
    outputs_d = X
    # create decoder model
    decoder = keras.Model(inputs_d, outputs_d)

    # Create autoencoder model
    auto = keras.Model(inputs_e, outputs_d)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
