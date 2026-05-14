#!/usr/bin/env python3
"""Write a function def cat_matrices(mat1, mat2, axis=0):
that concatenates two matrices along a specific axis:

    You can assume that mat1 and mat2 are matrices containing
    ints/floats
    You can assume all elements in the same dimension are
    of the same type/shape
    You must return a new matrix
    If you cannot concatenate the matrices, return None
    You can assume that mat1 and mat2 are never empty

    """


def cat_matrices(mat1, mat2, axis=0):
    """function def cat_matrices(mat1, mat2, axis=0):
that concatenates two matrices along a specific axis

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.
    """

    out = []
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    if axis == 0:
        out = mat1 + mat2
        return out

    if axis == 1:
        for i in range(len(mat1)):
            for j in range(len(mat2[i])):
                mat1[i].append(mat2[i][j])
        out.extend(mat1)
    
    if axis == 2:
        for i in range(len(mat1)):
            for j in range(len(mat2[i])):
                for k in range(len(mat2[i][j])):
                    mat1[i][j].append(mat2[i][j][k])
            out.extend(mat1[i])
        return out
