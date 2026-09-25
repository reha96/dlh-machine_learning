#!/usr/bin/env python3
# (Based on N/A — Task 0 has no preceding task skeleton.)
"""Create a vanilla autoencoder model."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a vanilla autoencoder.

    Args:
        input_dims: An integer containing the dimensions of the model input.
        hidden_layers: A list containing the number of nodes for each hidden
            layer in the encoder, respectively. The hidden layers should be
            reversed for the decoder.
        latent_dims: An integer containing the dimensions of the latent space
            representation.

    Returns:
        encoder, decoder, auto: The encoder model, decoder model, and full
            autoencoder model.

    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss. All layers should use a relu activation except
    for the last layer in the decoder, which should use sigmoid.
    """

    # This is our input
    input_img = keras.Input(shape=input_dims)

    # Encoder: compress input to latent space
    encoded = input_img
    for h in hidden_layers:
        encoded = keras.layers.Dense(h, activation='relu')(encoded)

    # Latent space representation (bottleneck)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    # Create encoder model (input → latent space)
    encoder = keras.Model(input_img, latent)

    # Decoder: expand latent space back to input dimensions
    decoded = latent
    for h in reversed(hidden_layers):
        decoded = keras.layers.Dense(h, activation='relu')(decoded)

    # Final output layer (sigmoid for reconstruction)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Create autoencoder model
    autoencoder = keras.Model(input_img, decoded)

    # Compile the autoencoder with Adam optimizer and binary cross-entropy loss
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Create decoder model (encoder + bottleneck + decoder)
    decoder = keras.Model(encoded, decoded)

    return encoder, decoder, autoencoder
