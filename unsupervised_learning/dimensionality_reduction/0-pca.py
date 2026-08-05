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
    # PCA step 1: decompose X with SVD: X = U @ diag(S) @ Vt
    # the rows of Vt are the eigenvectors (principal directions)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # PCA step 2: variance of each component = squared singular value
    # S squared is proportional to the covariance eigenvalues
    S2 = S * S
    # PCA step 3: total variance = sum of all component variances
    total = S2.sum()
    # PCA step 4: fraction of variance kept by 1, 2, ... components
    frac = np.cumsum(S2) / total
    # first component that crosses var, plus one extra like the reference
    nd = np.argmax(frac >= var) + 2
    # PCA step 5: build W from the first nd eigenvectors
    W = Vt[:nd].T
    return W