#!/usr/bin/env python3
"""Write a function that calculates the expectation step
in the EM algorithm for a GMM:
    """
import numpy as np


def expectation(X, pi, m, S):
    """    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors

    m is a numpy.ndarray of shape (k, d) containing the centroid means
    S is a numpy.ndarray of shape (k, d, d)
    containing the covariance matrices

    You may use at most 1 loop

    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities
        l is the total log likelihood
    You should use pdf = __import__('5-pdf').pdf
    """

