#!/usr/bin/env python3
"""
Write a function def normalization_constants(X): that calculates
the normalization (standardization) constants of a matrix
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix

    X is the numpy.ndarray of shape (m, nx) to normalize
    m is the number of data points
    nx is the number of features

    Returns: the mean and standard deviation of each feature,
    respectively
    """
    return np.mean(X, axis=0), np.std(X, axis=0)


if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(m)
    print(s)
