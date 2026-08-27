#!/usr/bin/env python3
"""
Write a function def update_variables_Adam(alpha, beta1, beta2, epsilon,
var, grad, v, s, t): that updates a variable in place using the Adam
optimization algorithm
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero

    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var
    s is the previous second moment of var
    t is the time step used for bias correction

    Returns: the updated variable, the new first moment, and the new
    second moment, respectively
    """

    # update v, which is the mean of the gradients
    v = beta1*v - (1-beta1)*grad

    # update s, which is the variance of the gradients
    s = beta2*s + (1-beta2)*np.square(grad)

    # correct v and s biases toward 0
    v = v/(1-beta1**t)
    s = s/(1-beta2**t)

    # update var, the parameter of interest
    var = var + alpha*v / (np.sqrt(s) + epsilon)

    return var, v, s