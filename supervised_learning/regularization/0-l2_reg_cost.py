#!/usr/bin/env python3
"""L2 regularization cost (numpy)."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    cost is a float containing the cost of the network without
    L2 regularization.
    lambtha is the regularization parameter.
    weights is a dictionary of the weights and biases (numpy.ndarrays)
    of the neural network.
    L is the number of layers in the neural network.
    m is the number of data points used.

    Returns: the cost of the network accounting for L2 regularization.
    """

    # cost is original loss plus penalty
    # lambtha is the strength of regularization param

    # first we need the penalty
    # penalty is the sum of the squares of the model's Weights
    # (per layer) but not Biases

    # create dict keys
    dkeys = []
    for i in range(1, L+1):
        dkeys.append(f"W{i}")

    # calculate sigma squared (sum of squared weights)
    sigma2 = 0
    for l in dkeys:
        # sum of squared matrices Layer X Weights (Frobenius**2)
        # vs regression feature weights (single vector)
        sigma2 += np.sum(weights[l]**2)

    # Why /2m? 1/m averages over examples
    # /2 cancels when you differentiate
    reg_cost = cost + (lambtha/(2*m))*sigma2
    return reg_cost
