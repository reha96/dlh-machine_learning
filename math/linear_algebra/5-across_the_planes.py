#!/usr/bin/env python3
"""Write a function def add_matrices2D(mat1, mat2): that adds two
matrices element-wise:

    You can assume that mat1 and mat2 are 2D matrices containing
    ints/floats
    You can assume all elements in the same dimension are of the
    same type/shape
    You must return a new matrix
    If mat1 and mat2 are not the same shape, return None

    """


def add_matrices2D(mat1, mat2):
    """function def add_matrices2D(mat1, mat2): that adds two
matrices element-wise

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    out = [[0] * len(mat1[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            out[i][j] = (mat1[i][j] + mat2[i][j])
    return out
