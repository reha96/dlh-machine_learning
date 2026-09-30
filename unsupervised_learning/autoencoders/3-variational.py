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

    # encoder
    X = keras.Input(input_dims)
    inputs_e = X

    for h in hidden_layers:
        X = keras.layers.Dense(h, activation='relu')(X)

    # create mean and var layer, log var is better for training
    z_mean = keras.layers.Dense(latent_dims, activation=None)(X)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(X)
    # Lambda layer allows arbitrary expressions as a Layer
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(inputs=inputs_e, outputs=[z, z_mean, z_log_var])

    # decoder
    inputs_d = keras.Input(shape=latent_dims)
    X = inputs_d
    for units in reversed(hidden_layers[1:]):  # Skip first element
        X = keras.layers.Dense(units, activation='relu')(X)
    outputs_d = keras.layers.Dense(input_dims, activation='sigmoid')(X)
    decoder = keras.Model(inputs=inputs_d, outputs=outputs_d)

    # auto
    # reconstruction loss needs to be added
    reconstructed = decoder(z)
    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs_e, reconstructed)
    reconstruction_loss *= input_dims

    # KL divergence loss
    kl_loss = -0.5 * keras.backend.sum(1 + z_log_var - keras.backend.square(
        z_mean) - keras.backend.exp(z_log_var), axis=-1)

    # Add combined loss to model
    auto_outputs = decoder(encoder(inputs_e))
    auto = keras.Model(inputs_e, auto_outputs)
    auto.add_loss(keras.backend.mean(reconstruction_loss + kl_loss))
    auto.compile(optimizer='adam')

    return encoder, decoder, auto


def sampling(args):
    """Reparameterization helper function for VAE."""
    z_mean, z_log_var = args  # unpack
    # std normal error
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(z_mean))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon
