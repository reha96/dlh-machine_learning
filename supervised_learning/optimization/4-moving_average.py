#!/usr/bin/env python3
"""
Write a function def moving_average(data, beta): that calculates the
weighted moving average of a data set
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average

    Your moving average calculation should use bias correction

    Returns: a list containing the moving averages of data
    """
    # create sliding window to avoid bias
    out = []
    v = 0
    # sliding window gives past data weight beta
    # new data gets 1-beta weight
    for i, d in enumerate(data):
        # d is each data point at i
        v = beta * v + (1-beta)*d
        # exponential correction to properly weigh each obs
        correction = (1 - beta**(i+1))
        out.append(v/correction)
    return out
