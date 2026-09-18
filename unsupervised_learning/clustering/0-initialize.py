#!/usr/bin/env python3
"""Write a function that initializes cluster centroids for K-means:
    """

import numpy as np


def initialize(X, k):
    """ X is a numpy.ndarray of shape (n, d) containing the dataset
    that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate
    uniform distribution along each dimension in d:
        The minimum values for the distribution should be the minimum
        values of X along each dimension in d
        The maximum values for the distribution should be the maximum
        values of X along each dimension in d
        You should use numpy.random.uniform exactly once
    You are not allowed to use any loops
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure

    """
    if not isinstance(X, np.ndarray) or \
            not isinstance(k, int) or k <= 0:
        return None
    else:
        try:
            d = X.shape[1]
            mins = X.min(axis=0)          # shape (d,) min of each column
            maxs = X.max(axis=0)          # shape (d,) max of each column
            centroids = np.random.uniform(mins, maxs, size=(k, d))
            return centroids
        except Exception:
            return None


import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()
    print(initialize(X, 5))