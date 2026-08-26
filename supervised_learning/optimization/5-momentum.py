#!/usr/bin/env python3
"""
Write a function def update_variables_momentum(alpha, beta1, var, grad,
v): that updates a variable using the gradient descent with momentum
optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum
    optimization algorithm

    alpha is the learning rate
    beta1 is the momentum weight

    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var

    Returns: the updated variable and the new moment, respectively
    """
    # first moment is calculated by the weighted average from previous task
    # we now apply it to the gradient
    # uncorrected to start small and build up eventually
    v = beta1*v + (1-beta1)*grad

    # minimize error rate with small steps
    var = var - alpha*v

    return (var, v)
