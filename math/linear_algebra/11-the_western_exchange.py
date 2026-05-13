#!/usr/bin/env python3
"""Write a function def np_transpose(matrix): that transposes
matrix:

    You can assume that matrix can be interpreted as a
    numpy.ndarray
    You are not allowed to use any loops or conditional
    statements
    You must return a new numpy.ndarray

    """


def np_transpose(matrix):
    """function def np_transpose(matrix): that transposes
matrix

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    out = matrix.transpose()
    return out
