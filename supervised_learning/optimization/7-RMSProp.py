#!/usr/bin/env python3
"""
Write a function def update_variables_RMSProp(alpha, beta2, epsilon, var,
grad, s): that updates a variable using the RMSProp optimization
algorithm
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero

    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var

    Returns: the updated variable and the new moment, respectively
    """

    # update the parameter vector using a decay rate beta2
    s = s * beta2 + (1 - beta2) * np.square(grad)

    # dynamically adjusting learning rate, Root Mean Square Propagation
    var = var - (alpha * grad) / np.sqrt(s + epsilon)

    return var, s
