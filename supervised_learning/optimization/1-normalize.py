#!/usr/bin/env python3
"""
Write a function def normalize(X, m, s): that normalizes (standardizes)
a matrix
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    X is the numpy.ndarray of shape (d, nx) to normalize
    d is the number of data points
    nx is the number of features

    m is a numpy.ndarray of shape (nx,) that contains the mean of all
    features of X
    s is a numpy.ndarray of shape (nx,) that contains the standard
    deviation of all features of X

    Returns: The normalized X matrix
    """
    mu = np.mean(X, axis=0) 
    sigma = np.std(X, axis=0)
    


if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(X[:10])
    X = normalize(X, m, s)
    print(X[:10])
    m, s = normalization_constants(X)
    print(m)
    print(s)