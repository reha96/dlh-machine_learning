#!/usr/bin/env python3
"""Write a function def cat_matrices2D(mat1, mat2, axis=0): that concatenates two matrices along a specific axis:

    You can assume that mat1 and mat2 are 2D matrices containing
    ints/floats
    You can assume all elements in the same dimension are of the
    same type/shape
    You must return a new matrix
    If the two matrices cannot be concatenated, return None

    """


def cat_matrices2D(mat1, mat2, axis=0):
    """a function def cat_matrices2D(mat1, mat2, axis=0):
    that concatenates two matrices along a specific axis

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.
    """

    if len(mat1) != len(mat2[0]) or axis > 1:
        return None
    out = []
    if axis == 0:
        out = mat1 + mat2
        return out

    if axis == 1:
        for i in range(len(mat1)):
            for j in range(len(mat2[i])):
                mat1[i].append(mat2[i][j])
        out.append(mat1)
        return out
