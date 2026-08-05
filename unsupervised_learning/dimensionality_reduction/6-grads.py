#!/usr/bin/env python3
"""Write a function that calculates the gradients of Y
    """
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional
transformation of X
P is a numpy.ndarray of shape (n, n) containing the P affinities

Returns:
dY, a numpy.ndarray of shape (n, ndim) containing the gradients of Y
Q, a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """
    n, ndim = Y.shape
    # step 1: get the Q affinities and their numerator
    Q, num = Q_affinities(Y)
    # step 2: how much the two distributions differ at each pair
    PQ = P - Q
    # step 3: dY[i] = sum over j of PQ(j,i) * num(j,i) * (Y[i] - Y[j])
    dY = np.zeros((n, ndim))
    for i in range(n):
        dY[i] = np.sum(
            (PQ[:, i, np.newaxis] * num[:, i, np.newaxis]) * (Y[i] - Y),
            axis=0)

    return dY, Q
