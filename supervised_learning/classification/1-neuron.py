#!/usr/bin/env python3
"""
Write a class that defines a single neuron performing
binary classification (Based on 0-neuron.py)
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
