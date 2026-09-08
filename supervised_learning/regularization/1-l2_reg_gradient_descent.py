#!/usr/bin/env python3
"""Gradient descent with L2 regularization (numpy)."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Y is a one-hot numpy.ndarray of shape (classes, m) containing the
    correct labels for the data.
    weights is a dictionary of the weights and biases of the neural
    network.
    cache is a dictionary of the outputs of each layer of the neural
    network.
    alpha is the learning rate.
    lambtha is the L2 regularization parameter.
    L is the number of layers of the network.

    Updates weights in place. Returns: None.
    """
    # gradient descent with L2 is first shrink then step
    # alpha is step size, lambtha is shrinkage strength (same as in cost)
    # we update W with penalty, without Biases

    # first we need m = number of examples
    m = Y.shape[1]

    # derivative of the cost function wrt
    # the final layer's pre-activation (dZ)
    # is the network's output (AL) - true labels (Y)
    dZ = cache['A'+str(L)] - Y

    # error must be propagated backward
    # starting from the output layer (L)
    for layer in range(L, 0, -1):
        # get activations, weights
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]

        # compute gradients
        # dW gets L2 term, db does not bc biases not regularized
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            # propagate to previous layer using original W
            # tanh derivative is 1 - A**2
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.square(A_prev))

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
