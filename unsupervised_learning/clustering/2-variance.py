#!/usr/bin/env python3
"""
Write a function that calculates the total intra-cluster
variance for a data set:
"""

import numpy as np


def variance(X, C):
    """X is a numpy.ndarray of shape (n, d) containing the data set
C is a numpy.ndarray of shape (k, d) containing the centroid
means for each cluster

You are not allowed to use any loops

Returns: var, or None on failure
var is the total variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    # re-compute clss when new C (if C changed)
    diffs = X[:, np.newaxis, :] - C
    # sum of the smallest squared distance across clusters
    var = (diffs ** 2).min(axis=1).sum()
    return var
