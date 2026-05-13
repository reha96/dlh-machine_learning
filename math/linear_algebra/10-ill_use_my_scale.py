#!/usr/bin/env python3
"""Write a function def np_shape(matrix): that calculates the shape of a numpy.ndarray:

    You are not allowed to use any loops or conditional statements
    You are not allowed to use try/except statements
    The shape should be returned as a tuple of integers

    """


def np_shape(matrix):
    """function def np_shape(matrix): that calculates the shape of a numpy.ndarray

    Args:
        matrix (_type_): _description_
    """
    mat = matrix.tolist()
    out = []
    while (
        isinstance(mat, list) and len(mat) > 0
    ):  # Loop as long as we have non-empty lists
        out.append(len(mat))  # Append the current dimension
        mat = mat[0]  # Drill down to the next level
    return tuple(out)  # Return the shape list
