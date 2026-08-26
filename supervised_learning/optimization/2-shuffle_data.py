#!/usr/bin/env python3
"""
Write a function def shuffle_data(X, Y): that shuffles the data points
in two matrices the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    X is the first numpy.ndarray of shape (m, nx) to shuffle
    m is the number of data points
    nx is the number of features in X

    Y is the second numpy.ndarray of shape (m, ny) to shuffle
    m is the same number of data points as in X
    ny is the number of features in Y

    Returns: the shuffled X and Y matrices
    """
    # both X and Y have the same nb of data points
    m = X.shape[0]
    # apply perm (an index array) to X and Y
    perm = np.random.permutation(m)
    return X[perm], Y[perm]
