#!/usr/bin/env python3
"""Write a function that finds the best number of clusters for a GMM
using the Bayesian Information Criterion:
    """

import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
X is a numpy.ndarray of shape (n, d) containing the data set
kmin is a positive integer containing the minimum number of clusters
kmax is a positive integer containing the maximum number of clusters

If kmax is None, kmax should be set to the maximum number of clusters
iterations is a positive integer containing the maximum iterations
tol is a non-negative float containing the tolerance
verbose is a boolean that determines if the EM algorithm should print

You may use at most 1 loop

Returns: best_k, best_result, l, b, or None, None, None, None on failure

best_k is the best value for k based on its BIC
best_result is tuple containing pi, m, S
pi is a numpy.ndarray of shape (k,) containing the cluster priors
m is a numpy.ndarray of shape (k, d) containing the centroid means
S is a numpy.ndarray of shape (k, d, d) containing the covariance
ll is a numpy.ndarray of shape (kmax - kmin + 1) containing the ll
b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC

Use: BIC = p * ln(n) - 2 * ll
p is the number of parameters required
n is the number of data points used to create the model
ll is the log likelihood of the model

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    elif not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    try:
        expectation_maximization = __import__('8-EM').expectation_maximization
        n, d = X.shape
        results = []
        ll = np.zeros(kmax - kmin + 1)
        b = np.zeros(kmax - kmin + 1)

        # we need k and i
        for i, k in enumerate(range(kmin, kmax + 1)):
            # run EM GMM
            pi, m, S, g, ll[i] = expectation_maximization(
                X, k, iterations, tol, verbose)
            # store all results
            results.append((pi, m, S))
            # store BIC
            p = (k-1) + k*d + k*d * (d + 1)/2
            b[i] = p * np.log(n) - 2 * ll[i]

        # find min BIC
        best_k = 1 + np.argmin(b)  # add 1 bc we start with kmin=1 + idx
        best_result = results[best_k-1]  # minus 1 bc we need idx only

        return best_k, best_result, ll, b
    except Exception:
        return None, None, None, None
