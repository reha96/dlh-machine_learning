#!/usr/bin/env python3
"""
Write a class DeepNeuralNetwork that defines a deep neural network
performing binary classification (Based on 15-neural_network.py)
"""

import numpy as np


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork that defines a deep neural network performing
    binary classification

    nx is the number of input features
    If nx is not an integer, raise a TypeError with the exception:
    nx must be an integer
    If nx is less than 1, raise a ValueError with the exception:
    nx must be a positive integer

    layers is a list representing the number of nodes in each layer
    of the network
    If layers is not a list or an empty list, raise a TypeError with
    the exception: layers must be a list of positive integers
    The first value in layers represents the number of nodes in the
    first layer
    If the elements in layers are not all positive integers, raise a
    TypeError with the exception: layers must be a list of positive
    integers

    All exceptions should be raised in the order listed above

    Public instance attributes:
    L: The number of layers in the neural network
    cache: A dictionary to hold all intermediary values of the network.
    Upon instantiation, it should be set to an empty dictionary
    weights: A dictionary to hold all weights and biases of the network.
    Upon instantiation, the weights should be initialized using the
    He et al. method and saved in the weights dictionary using the key
    Wl where l is the hidden layer the weight belongs to
    The biases should be initialized to 0's and saved in the weights
    dictionary using the key bl where l is the hidden layer the bias
    belongs to

    You are allowed to use one loop
    """

    def __init__(self, nx, layers):
        """
        class constructor

        nx is the number of input features
        layers is a list representing the number of nodes in each layer
        of the network

        Sets the public instance attributes:
        L: The number of layers in the neural network
        cache: A dictionary to hold all intermediary values of the
        network. Upon instantiation, it should be set to an empty
        dictionary
        weights: A dictionary to hold all weights and biases of the
        network. Upon instantiation, the weights should be initialized
        using the He et al. method and saved using the key Wl and
        biases initialized to 0's and saved using the key bl

        You are allowed to use one loop
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # layers instead of nodes
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # all layers elements are positive ints
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        # set three public attrs
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # initialize weights
        for l in range(self.L):
            # n_prev = nx in first layer, else previous layer size
            if l == 0:
                n_prev = nx
            else:
                n_prev = layers[l - 1]
            # He et al.: N(0,1) * sqrt(2 / n_prev)
            self.weights["W{}".format(
                l + 1)] = np.random.randn(layers[l], n_prev) * np.sqrt(2 / n_prev)
            self.weights["b{}".format(l + 1)] = np.zeros((layers[l], 1))
