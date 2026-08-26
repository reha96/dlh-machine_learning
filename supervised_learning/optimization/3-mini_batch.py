#!/usr/bin/env python3
"""
Write a function def create_mini_batches(X, Y, batch_size): that creates
mini-batches to be used for training a neural network using mini-batch
gradient descent
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network using
    mini-batch gradient descent

    X is a numpy.ndarray of shape (m, nx) representing input data
    m is the number of data points
    nx is the number of features in X

    Y is a numpy.ndarray of shape (m, ny) representing the labels
    m is the same number of data points as in X
    ny is the number of classes for classification tasks

    batch_size is the number of data points in a batch

    Returns: list of mini-batches containing tuples (X_batch, Y_batch)

    Your function should allow for a smaller final batch (i.e. use the
    entire dataset)
    """
    # shuffle first
    X, Y = shuffle_data(X, Y)

    # get nb of data points
    m = X.shape[0]

    # find number of batches given batch_size
    if m % batch_size != 0:
        n_batches = m // batch_size + 1
    else:
        n_batches = m // batch_size

    out = []
    for i in n_batches:
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        out.append((X_batch, Y_batch))

    return out
