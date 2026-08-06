#!/usr/bin/env python3
"""
Write a class that defines a single neuron performing
binary classification (Based on 3-neuron.py)
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
W: The weights vector. Upon instantiation,
it should be initialized using a random normal distribution.
b: The bias. Upon instantiation,
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

    def cost(self, Y, A):
        """
Calculates the cost of the model using logistic regression

Y is a numpy.ndarray with shape (1, m)
that contains the correct labels

A is a numpy.ndarray with shape (1, m)
containing the activated output of the neuron

To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
Returns the cost
        """
        try:
            # calculate cost, log opposite (1-prediction) of predictions
            # start with m extracting m examples to later average cost:
            m = Y.shape[1]
            # -1/m averages and changes sign of negative output
            cost = (-1/m) * np.sum((Y * np.log(A)) +
                                   ((1 - Y) * np.log(1.0000001 - A)))
            return cost
        except Exception:
            print("check input dims")
