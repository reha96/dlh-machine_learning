#!/usr/bin/env python3
"""Write a function that calculates the cost of a t-SNE transformation
    """
import numpy as np


def cost(P, Q):
    """
P is a numpy.ndarray of shape (n, n) containing the P affinities
Q is a numpy.ndarray of shape (n, n) containing the Q affinities

Returns:
C, the cost of the transformation
    """
    # step 1: avoid log of 0 and division by 0 with a small floor
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    # step 2: KL divergence of P against Q
    C = np.sum(P * np.log(P / Q))

    return C
