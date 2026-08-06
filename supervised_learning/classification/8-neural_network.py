#!/usr/bin/env python3
"""
Write a class that defines a neural network with one hidden layer
performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """
nx is the number of input features
If nx is not an integer, raise a TypeError with the exception:
nx must be an integer
If nx is less than 1, raise a ValueError with the exception:
nx must be a positive integer

nodes is the number of nodes in the hidden layer
If nodes is not an integer, raise a TypeError with the exception:
nodes must be an integer
If nodes is less than 1, raise a ValueError with the exception:
nodes must be a positive integer

All exceptions should be raised in the order listed above
    """

    def __init__(self, nx, nodes):
        """
class constructor

Public instance attributes:

W1: The weights vector for the hidden layer.
Upon instantiation, it should be initialized using a random normal
distribution.
b1: The bias for the hidden layer. Upon instantiation,
it should be initialized to 0.
A1: The activated output for the hidden layer. Upon instantiation,
it should be initialized to 0.

W2: The weights vector for the output neuron. Upon instantiation,
it should be initialized using a random normal distribution.
b2: The bias for the output neuron. Upon instantiation,
it should be initialized to 0.
A2: The activated output for the output neuron (prediction).
Upon instantiation, it should be initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # nodes checks
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # init neuron "Hidden Layer" x nodes
        # weights (W1) is of shape (nodes, nx), drawing from std normal
        self.W1 = np.random.standard_normal(size=(nodes, nx))
        # neutral bias (b) init
        self.b1 = np.zeros((nodes, 1))
        # neuron answer (A) init
        self.A1 = 0
        # init neuron 2 "Output Layer"
        # weights (W1) is of shape (1, nodes), drawing from std normal
        self.W2 = np.random.standard_normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
