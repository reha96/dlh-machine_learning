#!/usr/bin/env python3
"""Write a function def np_slice(matrix, axes={}): that
slices a matrix along specific axes:

    You can assume that matrix is a numpy.ndarray
    You must return a new numpy.ndarray
    axes is a dictionary where the key is an axis to slice
    along and the value is a tuple representing the slice
    to make along that axis
    You can assume that axes represents a valid slice
    """


import numpy as np


def np_slice(matrix, axes={}):
    keys = []
    temp = [slice(None)] * matrix.ndim
    for i in axes.keys():
        keys.append(i)
    for key in keys:
        temp[key] = axes.get(key)
        temp[key] = slice(*temp[key])
    return matrix[tuple(temp)]
