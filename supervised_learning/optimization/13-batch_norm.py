#!/usr/bin/env python3
"""
Write a function def batch_norm(Z, gamma, beta, epsilon): that normalizes
an unactivated output of a neural network using batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization

    Z is a numpy.ndarray of shape (m, n) that should be normalized
    m is the number of data points
    n is the number of features in Z
    gamma is a numpy.ndarray of shape (1, n) containing the scales used
    for batch normalization
    beta is a numpy.ndarray of shape (1, n) containing the offsets used
    for batch normalization
    epsilon is a small number to avoid division by zero

    Returns: the normalized Z matrix
    """
    # standardize as usual
    Z_std = (Z-np.mean(Z, axis=0))/(np.std(Z+epsilon, axis=0))

    # now normalize using gamma and beta
    Z_norm = Z_std * gamma + beta

    return Z_norm
