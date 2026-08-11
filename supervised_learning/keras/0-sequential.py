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
    model = K.Sequential()
    for i in range(len(layers)):
        model.add(K.layers.Dense(
            layers[i], input_dim=nx, activation=activations[i], kernel_regularizer=K.regularizers.L2(lambtha)))
        if keep_prob is not None:
            model.add(K.layers.Dropout(rate=1-keep_prob))

    return model
