#!/usr/bin/env python3
"""Write a function that performs PCA on a dataset
    """
import numpy as np


def pca(X, var=0.95):
    """
X is a numpy.ndarray of shape (n, d) where:
n is the number of data points
d is the number of dimensions in each point
all dimensions have a mean of 0 across all data points
var is the fraction of the variance that the PCA should maintain

Returns:
the weights matrix, W, that maintains var fraction of X's variance
W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    n, d = X.shape  # with standardized means so no additional centering
    # PCA step 1: covariance mat where Xs are distances from 0
    cov = (1/n) * (X.T @ X)
    # PCA step 2: eigendecompose covariance matrix 
    evals, evecs = np.linalg.eigh(cov)
    # 
    return W


np.random.seed(0)
a = np.random.normal(size=50)
b = np.random.normal(size=50)
c = np.random.normal(size=50)
d = 2 * a
e = -5 * b
f = 10 * c

X = np.array([a, b, c, d, e, f]).T
m = X.shape[0]
X_m = X - np.mean(X, axis=0)
W = pca(X_m)
T = np.matmul(X_m, W)
print(T)
X_t = np.matmul(T, W.T)
print(np.sum(np.square(X_m - X_t)) / m)
