#!/usr/bin/env python3
"""Write a function that calculates the expectation step
in the EM algorithm for a GMM:
    """
import numpy as np


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors

    m is a numpy.ndarray of shape (k, d) containing the centroid means
    S is a numpy.ndarray of shape (k, d, d)
    containing the covariance matrices

    You may use at most 1 loop

    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities
        l is the total log likelihood
    """
    
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2 or \
            S.shape[0] != S.shape[1]:
        return None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None
    try:
        pdf = __import__('5-pdf').pdf
        
        # d dimensional
        d = X.shape[1]

        # multivariate normal pdf formula
        const = (2*np.pi)**(-d/2)*(np.linalg.det(S))**(-1/2)
        # (X - m) has shape (n, d), inv(S) has shape (d, d), sum along d
        mahl = -1/2*(((X-m) @ np.linalg.inv(S) * (X-m)).sum(axis=1))
        P = np.maximum(const * np.exp(mahl), 10**(-300))
        return P
    except Exception:
        return None