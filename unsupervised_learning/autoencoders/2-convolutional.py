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

    # Build the encoder.
    encoder_input = keras.Input(shape=input_dims)
    encoded = encoder_input
    for f in filters:
        encoded = keras.layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding='same',
            activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D(
            pool_size=(2, 2), padding='same')(encoded)
    encoder = keras.Model(encoder_input, encoded, name='encoder')

    # Build the decoder.
    decoder_input = keras.Input(shape=latent_dims)
    decoded = decoder_input
    for f in reversed(filters[1:]):
        decoded = keras.layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding='same',
            activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)

    decoded = keras.layers.Conv2D(
        filters=filters[0], kernel_size=(3, 3), padding='valid',
        activation='relu')(decoded)
    decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        filters=input_dims[-1], kernel_size=(3, 3), padding='same',
        activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoded, name='decoder')

    # Connect the encoder and decoder.
    auto_input = keras.Input(shape=input_dims)
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
