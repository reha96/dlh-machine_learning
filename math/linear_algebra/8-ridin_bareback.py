#!/usr/bin/env python3
"""Write a function def mat_mul(mat1, mat2): that performs matrix
multiplication:

    You can assume that mat1 and mat2 are 2D matrices containing
    ints/floats
    You can assume all elements in the same dimension are of the
    same type/shape
    You must return a new matrix
    If the two matrices cannot be multiplied, return None

    """


def mat_mul(mat1, mat2):
    """function def mat_mul(mat1, mat2): that performs matrix
multiplication

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
    """

    rows = range(len(mat2))
    cols = range(len(mat1[0]))
    if rows != cols:
        return None
    out = []
    for i in range(len(mat1)):
        temp_sum = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in rows:
                sum += mat1[i][k] * mat2[k][j]
            temp_sum.append(sum)
        out.append(temp_sum)
    return out
