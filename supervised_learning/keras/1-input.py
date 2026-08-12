#!/usr/bin/env python3
"""Builds a neural network with the Keras library."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx (int): the number of input features to the network
        layers (list): containing the number of nodes in each layer of the
            network
        activations (list): containing the activation functions used for each
            layer of the network
        lambtha (float): the L2 regularization parameter
        keep_prob (float): the probability that a node will be kept for
            dropout

    Returns:
        the keras model
    """
    # define incoming features
    inputs = K.Input(shape=(nx,))
    # assign (reference to the same tensor)
    x = inputs
    # one Dense per entry in layers, in order
    for i in range(len(layers)):
        # Dense: W·x + b then activation — W and b are created inside
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.L2(lambtha))(x)
        # dropout randomly silences nodes during training (1 - keep_prob
        # chance); never on the output layer
        if keep_prob is not None and i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)
    # input tensor, output tensor
    model = K.Model(inputs, x)
    return model
