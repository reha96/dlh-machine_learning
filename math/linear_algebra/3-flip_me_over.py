#!/usr/bin/env python3
"""Write a function def matrix_transpose(matrix): that returns the
transpose of a 2D matrix, matrix:

You must return a new matrix
You can assume that matrix is never empty
You can assume all elements in the same dimension are of the same
type/shape

"""


def matrix_transpose(matrix):
    """function def matrix_transpose(matrix): that returns the
    transpose of a 2D matrix, matrix

    Args:
        matrix (_type_): _description_
    """
    rows = range(len(matrix))
    cols = range(len(matrix[0]))
    out = []
    for i in cols:
        temp = []
        for j in rows:
            temp.append(matrix[j][i])
            print(temp)
        out.append(temp)
    return out
