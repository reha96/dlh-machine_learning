#!/usr/bin/env python3
"""
Write a function that calculates the total intra-cluster
variance for a data set:
"""

import numpy as np


def variance(X, C):
    """X is a numpy.ndarray of shape (n, d) containing the data set
C is a numpy.ndarray of shape (k, d) containing the centroid
means for each cluster

You are not allowed to use any loops

Returns: var, or None on failure
var is the total variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    # re-compute clss with new C (if C changed)
    diffs = X[:, np.newaxis, :] - C
    var = (diffs ** 2).sum(axis=2)
    # clss = dists.argmin(axis=1)
    return var

import numpy as np
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    for k in range(1, 11):
        C, _ = kmeans(X, k)
        print('Variance with {} clusters: {}'.format(k, variance(X, C).round(5)))