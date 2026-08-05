#!/usr/bin/env python3
"""Write a function that performs PCA on a dataset
    """
import numpy as np


def pca(X, ndim):
    """
X is a numpy.ndarray of shape (n, d) where:
n is the number of data points
d is the number of dimensions in each point

ndim is the new dimensionality of the transformed X

Returns:
T, a numpy.ndarray of shape (n, ndim) containing the transformed version of X
    """
    if not isinstance(ndim, int) or ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    # PCA step 1: center X so each feature has a mean of 0
    X_c = X - np.mean(X, axis=0)
    # PCA step 2: decompose centered X with SVD: X_c = U @ diag(S) @ Vt
    # the rows of Vt are the eigenvectors (principal directions)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    # PCA step 3: cap ndim at the number of components available
    ndim = min(ndim, Vt.shape[0])
    # PCA step 4: project centered data onto the first ndim directions
    T = X_c @ Vt[:ndim].T
    return T
