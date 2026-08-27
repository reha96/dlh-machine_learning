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
    mean = np.mean(Z, axis=0)  # axis=0, ddof=0 population
    variance = np.var(Z, axis=0)  # ddof=0 default, NOT np.std
    # epsilon INSIDE sqrt(var+epsilon)
    # adding constant inside np.std would cancel out, so above order
    Z_std = (Z - mean) / np.sqrt(variance + epsilon)

    # now normalize using gamma and beta
    Z_norm = Z_std * gamma + beta  # gamma,beta (1,n) broadcast

    return Z_norm
