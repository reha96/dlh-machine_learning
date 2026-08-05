#!/usr/bin/env python3
"""Write a function that initializes the P affinities for t-SNE
    """
import numpy as np


def P_init(X, perplexity):
    """
X is a numpy.ndarray of shape (n, d) containing the dataset
n is the number of data points
d is the number of dimensions in each point
perplexity is the perplexity that all Gaussian distributions should have

Returns:
D, a numpy.ndarray of shape (n, n) with the pairwise distances between points
P, a numpy.ndarray of shape (n, n) initialized to all 0s
betas, a numpy.ndarray of shape (n, 1) initialized to all 1s
H, the Shannon entropy for a discrete random variable with log base 2
    """
    n, d = X.shape
    # step 1: squared distance of each point from the origin
    sum_X = np.sum(np.square(X), axis=1)
    # step 2: pairwise squared distances = sum_X + sum_X.T - 2 * X X.T
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    # step 3: distance of each point to itself is 0
    np.fill_diagonal(D, 0)
    # step 4: P affinities start as all 0s
    P = np.zeros((n, n))
    # step 5: each Gaussian starts with precision beta = 1
    betas = np.ones((n, 1))
    # step 6: target entropy = log base 2 of the perplexity
    H = np.log2(perplexity)

    return D, P, betas, H
