#!/usr/bin/env python3
"""Write a function that calculates
a correlation matrix

"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix

    Args:
        C (_type_): _description_
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diagonal = np.diagonal(C)
    outer = np.diagflat(diagonal)**(-1/2)
    Co = outer @ C @ outer
    return Co
