#!/usr/bin/env python3
"""Write a function that calculates the Shannon entropy and P affinities
    """
import numpy as np


def HP(Di, beta):
    """
Di is a numpy.ndarray of shape (n - 1,) containing the pairwise distances
between a data point and all other points
beta is a numpy.ndarray of shape (1,) containing the beta value for the
Gaussian distribution

Returns:
Hi, the Shannon entropy of the points
Pi, a numpy.ndarray of shape (n - 1,) containing the P affinities
    """
    # step 1: Gaussian kernel weight for every distance
    num = np.exp(-Di * beta)
    # step 2: normalize the weights into probabilities
    Pi = num / np.sum(num)
    # step 3: Shannon entropy with log base 2, in a numerically stable form
    # that never takes log2 of a zero: Hi = log2(sum) + beta * E[Di] / ln(2)
    Hi = np.log2(np.sum(num)) + beta * np.sum(Di * Pi) / np.log(2)

    return Hi, Pi
