#!/usr/bin/env python3
"""Write a function that initializes variables for a
Gaussian Mixture Model:

    """

import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set

    k is a positive integer containing the number of clusters

    You are not allowed to use any loops

    Returns: pi, m, S, or None, None, None on failure

        pi is a numpy.ndarray of shape (k,) containing the priors
        for each cluster, initialized evenly

        m is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster, initialized with K-means

        S is a numpy.ndarray of shape (k, d, d) containing
        the covariance matrices for each cluster, initialized
        as identity matrices
    """

    if not isinstance(X, np.ndarray) or len(X.shape) < 2:
        return (None, None)

    kmeans = __import__('1-kmeans').kmeans
    n, d = X.shape
    m, clss = kmeans(X, k)
    # create new array with shape (k, ) and value
    pi = np.full(k, 1/k) # neutral prior, equally likely
    S = np.identity(d) # (d, d) identity matrix
    S = S[np.newaxis, :,:] # add 1 dimension (1, d, d)
    S = S.repeat(k, axis=0) # repeat k times along first axis 
    