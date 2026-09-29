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
    # This is our input
    input_img = keras.Input(shape=input_dims)

    # Encoder: compress input to latent space
    X = input_img
    for h in hidden_layers:
        X = keras.layers.Dense(h, activation='relu')(X)

    # Latent space representation (bottleneck)
    latent = keras.layers.Dense(latent_dims, activation='relu')(X)
    # Create encoder model (input → latent space)
    encoder = keras.Model(input_img, latent)

    # Decoder: expand latent space back to input dimensions
    latent_inputs = keras.Input(shape=(latent_dims,))
    X = latent_inputs
    for h in reversed(hidden_layers):
        X = keras.layers.Dense(h, activation='relu',
                               activity_regularizer\
                                   =keras.regularizers.l1(lambtha))(X)

    # Final output layer (sigmoid for reconstruction)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(X)
    # Create decoder model (latent space → output)
    decoder = keras.Model(latent_inputs, outputs)

    # Create autoencoder model
    auto_outputs = decoder(encoder(input_img))
    auto = keras.Model(input_img, auto_outputs)

    # Compile the autoencoder with Adam optimizer and binary cross-entropy loss
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
