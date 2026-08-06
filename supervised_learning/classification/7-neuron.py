#!/usr/bin/env python3
"""
Write a class that defines a single neuron performing
binary classification (Based on 6-neuron.py)
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
Evaluates the neuron's predictions

X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels

Returns the neuron's prediction and the cost of the network, respectively

The prediction should be a numpy.ndarray with shape (1, m)
containing the predicted labels

The label values should be 1 if the output of the network is >= 0.5
and 0 otherwise
        """
        try:
            # get predictions A from neuron
            A = self.forward_prop(X)
            # loss from A
            cost = self.cost(Y, A)
            # label predictions 1 and 0
            prediction = np.where(A >= 0.5, 1, 0)
            return prediction, cost
        except Exception:
            print("check input dims")

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
Calculates one pass of gradient descent on the neuron

X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels
A is a numpy.ndarray with shape (1, m) that contains the activated output

alpha is the learning rate, defaulting to 0.05

Updates the private attributes __W and __b
        """
        # residual (A - Y): prediction minus truth, signed mistake per example
        dz = A - Y
        # m is the number of examples (columns of X)
        m = X.shape[1]
        # derivative of cost w.r.t. W: avg of (mistake x input)
        dW = (1 / m) * np.matmul(dz, X.T)
        # derivative of cost w.r.t. b: avg mistake
        db = (1 / m) * np.sum(dz)
        # step downhill: subtract learning rate x gradient
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
Trains the neuron
X is a numpy.ndarray with shape (nx, m) that contains the input data

nx is the number of input features to the neuron
m is the number of examples

Y is a numpy.ndarray with shape (1, m) that contains the correct
labels for the input data

verbose is a boolean that defines whether or not to print
information about the training
graph is a boolean that defines whether or not to graph
information about the training

if verbose or graph is True:

if step is not an integer, raise a TypeError with the exception
step must be an integer

if step is not positive or greater than iterations,
raise a ValueError with the exception
step must be positive and <= iterations

All exceptions should be raised in the order listed above
Updates the private attributes __W, __b, and __A

You are allowed to use one loop

If verbose is True, print Cost after {iteration} iterations: {cost}
every step iterations, starting at 0

The graph should be plotted with plt.show()
Returns the evaluation of the training data after
iterations of training have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if not iterations > 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if not alpha > 0:
            raise ValueError("alpha must be positive")
        # new checks
        if verbose or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
        if not step > 0 or not step <= iterations:
            raise ValueError("step must be positive and <= iterations")

        for i in range(iterations):
            # make predictions from X
            A = self.forward_prop(X)
            # improve weight (W) and bias (b) from predictions A
            self.gradient_descent(X, Y, A, alpha)
            # update loop and report cost
            if verbose is True:
                cost = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost}")
        return self.evaluate(X, Y)
