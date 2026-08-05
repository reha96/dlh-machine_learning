#!usr/bin/env python3
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
p is the number of parameters required for the model
n is the number of data points used to create the model
ll is the log likelihood of the model

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if kmax is not None:
        kmax = X.shape[0]
    elif not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    expectation_maximization = __import__('8-EM').expectation_maximization
    try:
        best_k = 0
        best_result = (pi, m, S)
        b = 0
        n, d = X.shape
        k = 0
        for i in range(iterations):
            pi, m, S, g, ll = expectation_maximization(X, k)
            p = (k−1) + k·d + k·d(d+1)/2
            b = p * ln(n) - 2 * ll
        return best_k, best_result, ll, b
    except Exception:
        return None, None, None, None
