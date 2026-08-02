#!/usr/bin/env python3
"""Write a function that calculates the expectation step
in the EM algorithm for a GMM:
    """
import numpy as np


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors

    m is a numpy.ndarray of shape (k, d) containing the centroid means
    S is a numpy.ndarray of shape (k, d, d)
    containing the covariance matrices

    You may use at most 1 loop

    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities
        l is the total log likelihood
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3 or \
            S.shape[1] != S.shape[2]:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1 or \
            pi.sum(axis=0) != 1:
        return None, None
    try:
        pdf = __import__('5-pdf').pdf

        # initialize
        k = pi.shape[0]  # clusters
        n = X.shape[0]  # data points
        g = np.zeros((k, n))  # posterior starts with zeros (k, n)
        denominator = np.zeros(n)  # evidence starts with zeros (n)

        # bayes's rule, loop over k clusters: post = prior x likelihood
        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])
            denominator += g[i]  # incrementally add evidence

        g = g/denominator  # post = prior x likelihood / evidence
        l = np.sum(np.log(denominator))  # log likelihood

        return g, l

    except Exception:
        return None, None
