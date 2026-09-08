#!/usr/bin/env python3
"""Gradient descent with Dropout (numpy)."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Y is a one-hot numpy.ndarray of shape (classes, m) containing the
    correct labels for the data.
    weights is a dictionary of the weights and biases of the neural
    network.
    cache is a dictionary of the outputs and dropout masks of each layer
    of the neural network.
    alpha is the learning rate.
    keep_prob is the probability that a node will be kept.
    L is the number of layers of the network.

    Updates weights in place. Returns: None.
    """
    # get examples
    m = Y.shape[1]

    # last layer derivative
    dZ = cache[f'A{L}']-Y

    # GD = backprop
    for l in range(L, 0, -1):
        A_prev = cache[f'A{l-1}']
        W = weights[f'W{l}']

        # gradients = derivatives
        dW = (1/m)*np.matmul(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)

        # dropout
        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            D = cache[f'D{l-1}']
            dA_prev *= D
            dA_prev /= keep_prob
            dZ = dA_prev*(1-np.square(A_prev))

    weights[f'W{l}'] -= alpha*dW
    weights[f'b{l}'] -= alpha*db
