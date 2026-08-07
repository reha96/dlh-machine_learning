#!/usr/bin/env python3
"""
Write a class that defines a neural network with one hidden layer
performing binary classification (Based on 10-neural_network.py)
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

Private instance attributes:
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
        self.__W1 = np.random.standard_normal(size=(nodes, nx))
        # neutral bias (b) init
        self.__b1 = np.zeros((nodes, 1))
        # neuron answer (A) init
        self.__A1 = 0
        # init neuron 2 "Output Layer"
        # weights (W2) is of shape (1, nodes), drawing from std normal
        self.__W2 = np.random.standard_normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
Calculates the forward propagation of the neural network

X is a numpy.ndarray with shape (nx, m) that contains the input data

nx is the number of input features to the neuron
m is the number of examples

Updates the private attributes __A1 and __A2
The neuron should use a sigmoid activation function

Returns a tuple containing the private attributes __A1 and __A2, respectively
        """
        # calculate z1, weights x input plus bias
        z1 = np.matmul(self.__W1, X) + self.__b1
        # squeeze z1 using sigmoid function between (0, 1)
        self.__A1 = 1/(1+np.exp(-z1))

        # only dimensions change, but np broadcasting handles it
        # calculate z2, uses A1 as input!
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # squeeze z2 using sigmoid function between (0, 1)
        self.__A2 = 1/(1+np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
Calculates the cost of the model using logistic regression

Y is a numpy.ndarray with shape (1, m)
that contains the correct labels for the input data

A is a numpy.ndarray with shape (1, m)
containing the activated output of the neuron for each example

To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
Returns the cost
        """
        # calculate cost, same as 3-cost.py
        # start with m extracting m examples to later average cost:
        m = Y.shape[1]
        # -1/m averages and changes sign of negative output
        cost = (-1/m) * np.sum((Y * np.log(A)) +
                               ((1 - Y) * np.log(1.0000001 - A)))
        return cost
