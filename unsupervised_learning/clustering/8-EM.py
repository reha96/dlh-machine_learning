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

You should use:
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization
You may use at most 1 loop

Returns: pi, m, S, g, l, or None, None, None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the priors 
m is a numpy.ndarray of shape (k, d) containing the centroid means 
S is a numpy.ndarray of shape (k, d, d) containing the covs
g is a numpy.ndarray of shape (k, n) containing the probabilities
l is the log likelihood of the model
"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    
