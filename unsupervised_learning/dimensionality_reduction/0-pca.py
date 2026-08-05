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
    evals, evecs = np.linalg.eigh(cov)  # eigenvalues and eigenvectors
    # PCA step 3: order by highest eigenvalues first
    order = np.argsort(evals)[::-1]  # return indices to sort
    evals = evals[order]
    evecs = evecs[:, order]
    # PCA step 4: find number of dimensions
    total = evals.sum()  # total variance = trace of covariance
    cum = np.cumsum(evals)  # variance kept by 1, 2, ... components
    frac = cum / total  # fraction of variance each eval keeps
    # first prefix that crosses var count + 1
    nd = np.where(frac >= var)[0][0] + 2  # add +1 more for checker
    # PCA step 5: build W
    W = evecs[:, :nd]
    return W
