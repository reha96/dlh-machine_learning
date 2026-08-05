#!/usr/bin/env python3
"""Write a function that calculates the Q affinities
    """
import numpy as np


def Q_affinities(Y):
    """
Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional
transformation of X

Returns:
Q, a numpy.ndarray of shape (n, n) containing the Q affinities
num, a numpy.ndarray of shape (n, n) containing the numerator of the
Q affinities
    """
    # step 1: squared distance of each point from the origin
    sum_Y = np.sum(np.square(Y), axis=1)
    # step 2: pairwise squared distances = sum_Y + sum_Y.T - 2 * Y Y.T
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    # step 3: Student-t kernel: 1 / (1 + distance squared)
    num = 1 / (1 + D)
    # step 4: distance of each point to itself is 0
    np.fill_diagonal(num, 0)
    # step 5: normalize into probabilities
    Q = num / np.sum(num)

    return Q, num
