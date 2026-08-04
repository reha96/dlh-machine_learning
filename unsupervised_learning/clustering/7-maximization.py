#!/usr/bin/env python3
"""Write a function that calculates the maximization step in the EM
"""

import numpy as np


def maximization(X, g):
    """
X is a numpy.ndarray of shape (n, d) containing the data set

g is a numpy.ndarray of shape (k, n) containing the posterior

You may use at most 1 loop

Returns: pi, m, S, or None, None, None on failure

pi is a numpy.ndarray of shape (k,) containing the updated priors
m is a numpy.ndarray of shape (k, d) containing the updated means
S is a numpy.ndarray of shape (k, d, d) containing the updated covs
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if np.any(g < 0):  # posteriors are non-negative
        return None, None, None
    try:
        n, d = X.shape  # data points, dims
        k = g.shape[0]  # clusters
        m = np.zeros((k, d))  # initialize cluster means
        S = np.zeros((k, d, d))  # initialize cov matrix
        pi = np.zeros(k)  # intialize prior
        for i in range(k):
            nk = g[i].sum()  # soft count N_k - total weight of cluster
            pi[i] = nk / n  # soft share - cluster k's share of the data
            m[i] = g[i] @ X / nk    # weighted mean: (n,) @ (n, d)
            diff = X - m[i]  # (n, d) deviations, NEW mean
            S[i] = (g[i][:, np.newaxis] * diff).T @ diff / nk
        return pi, m, S
    except Exception:
        return None, None, None
