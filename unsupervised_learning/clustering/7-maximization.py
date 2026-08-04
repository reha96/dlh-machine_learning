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
    try:
        n, d = X.shape
                
    except Exception:
        return None, None, None
