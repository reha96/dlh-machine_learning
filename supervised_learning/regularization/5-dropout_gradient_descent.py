#!/usr/bin/env python3
"""Gradient descent with Dropout (numpy)."""
# Spec: intranet 2297 (2026-09-05). Stub only, no solution code.
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
    pass
