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
    model.add(input_shape=(nx,))
    for i in range(layers):
        model.add(K.layers.Dense(
            layers[i], activation=activations[i], bias_regularizer=lambtha))

    return model
