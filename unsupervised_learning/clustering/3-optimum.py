#!/usr/bin/env python3
"""Write a function that tests for the optimum number of clusters
by variance:
    """


import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """X is a numpy.ndarray of shape (n, d) containing the data set

kmin is a positive integer containing the minimum number of clusters
to check for (inclusive)

kmax is a positive integer containing the maximum number of clusters
to check for (inclusive)

iterations is a positive integer containing the maximum number of
iterations for K-means

This function should analyze at least 2 different cluster sizes

You may use at most 2 loops

Returns: results, d_vars, or None, None on failure
results is a list containing the outputs of K-means
d_vars is a list containing the difference in variance from
the smallest cluster size

    """
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or kmin <= 0 \
        or not isinstance(kmax, int) or kmax <= 0 \
            or not isinstance(iterations, int) or iterations <= 0:
        return (None, None)
    else:
        try:
            kmeans = __import__('1-kmeans').kmeans
            variance = __import__('2-variance').variance

            results = []
            var_list = []
            d_vars = []
            for i in range(kmin, kmax+1):
                C, clss = kmeans(X, i, iterations)
                results.append((C, clss))
                var = variance(X, C)
                var_list.append(var)

            var_baseline = var_list[0]
            for j in var_list:
                d_vars.append(var_baseline - j)

            return (results, d_vars)
        except Exception:
            return (None, None)
