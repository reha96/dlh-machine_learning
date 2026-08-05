#!/usr/bin/env python3
"""Write a function that performs the expectation maximization for a GMM
"""

import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
X is a numpy.ndarray of shape (n, d) containing the data set

k is a positive integer containing the number of clusters

iterations is a positive integer containing the maximum number
of iterations

tol is a non-negative float containing tolerance of the log likelihood,
used to determine early stopping i.e. if the difference is less than or
equal to tol you should stop the algorithm

verbose is a boolean that determines if you should print information
about the algorithm

If True, print Log Likelihood after {i} iterations: {l} every 10
iterations and after the last iteration
{i} is the number of iterations of the EM algorithm
{l} is the log likelihood, rounded to 5 decimal places

You may use at most 1 loop

Returns: pi, m, S, g, ll, or None, None, None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the priors
m is a numpy.ndarray of shape (k, d) containing the centroid means
S is a numpy.ndarray of shape (k, d, d) containing the covs
g is a numpy.ndarray of shape (k, n) containing the probabilities
ll is the log likelihood of the model
"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    try:
        initialize = __import__('4-initialize').initialize
        expectation = __import__('6-expectation').expectation
        maximization = __import__('7-maximization').maximization

        pi, m, S = initialize(X, k)
        # E step: soft assignments g (k, n) and model log-likelihood ll
        g, ll = expectation(X, pi, m, S)
        prev_ll = 0.0  # sentinel: |0 - ll| is huge, iteration 0 never stops

        for i in range(iterations):
            if verbose is True and i % 10 == 0:
                print(
                    f"Log Likelihood after {i} iterations: "
                    f"{np.round(ll, 5)}")
            # M step: new priors pi, means m, covariances S
            pi, m, S = maximization(X, g)
            # E step: recompute g and ll from the updated parameters
            g, ll = expectation(X, pi, m, S)
            # stop when the likelihood stops moving by more than tol
            if abs(ll - prev_ll) <= tol:
                break
            prev_ll = ll

        if verbose is True:
            print(
                f"Log Likelihood after {i + 1} iterations: "
                f"{np.round(ll, 5)}")

            return pi, m, S, g, ll
        # regular stopping after loop
        if verbose is True:
            print(
                f"Log Likelihood after {iterations} iterations: "
                f"{np.round(ll, 5)}")

        return pi, m, S, g, ll

    except Exception:
        return None, None, None, None, None
