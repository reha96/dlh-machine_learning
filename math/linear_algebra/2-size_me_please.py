#!/usr/bin/env python3
"""Write a function def matrix_shape(matrix): that calculates
the shape of a matrix
"""


def matrix_shape(mat):
    """a function def matrix_shape(matrix): that calculates
    the shape of a matrix

    Args:
        mat (_type_): _description_
    Returns:
        _type_: _description_
    """
    out = []
    while (
        isinstance(mat, list) and len(mat) > 0
    ):  # Loop as long as we have non-empty lists
        out.append(len(mat))  # Append the current dimension
        mat = mat[0]  # Drill down to the next level
    return out  # Return the shape list
