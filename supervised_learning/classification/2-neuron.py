#!/usr/bin/env python3
"""
Write a class that defines a single neuron performing
binary classification (Based on 1-neuron.py)
"""

import numpy as np


class Neuron:
    """
nx is the number of input features to the neuron
If nx is not an integer, raise a TypeError with the exception:
nx must be an integer
If nx is less than 1, raise a ValueError with the exception:
nx must be a positive integer
All exceptions should be raised in the order listed above
    """

    def __init__(self, nx):
        """
class constructor

Public instance attributes:
W: The weights vector for the neuron. Upon instantiation,
it should be initialized using a random normal distribution.
b: The bias for the neuron. Upon instantiation,
it should be initialized to 0.
A: The activated output of the neuron (prediction). Upon instantiation,
it should be initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # weights (W) is of shape (1, nx) or nx.T, drawing from std normal
        self.__W = np.random.standard_normal(size=(1, nx))
        # neutral bias (b) init
        self.__b = 0
        # neuron answer (A) init
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
Calculates the forward propagation of the neuron
X is a numpy.ndarray with shape (nx, m) that contains the input data

nx is the number of input features to the neuron
m is the number of examples

Updates the private attribute __A
The neuron should use a sigmoid activation function
Returns the private attribute __A
        """
        try:
            # calculate z, weights x input plus bias
            z = np.matmul(self.__W, X) + self.__b
            # squeeze z using sigmoid function between (0, 1)
            self.__A = 1/(1+np.exp(-z))
            return self.__A
        except Exception:
            print("check input dims")
