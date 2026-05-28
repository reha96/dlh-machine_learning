#!/usr/bin/env python3

"""Write a function def minor(matrix): that calculates the minor
matrix of a matrix:
    """


def minor(matrix):
    """calculates the minor
matrix of a matrix

    Args:
        matrix (_type_): _description_
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Every element must be a list
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    submatrix = []
    for col in range(len(matrix)):  # col = 0, 1, 2
        temp = []
        for row_idx in range(1, len(matrix)):  # skip first row (index 0)
            # remove the current column from this row
            new_row = matrix[row_idx][:col] + matrix[row_idx][col+1:]
            temp.append(new_row)
        submatrix.append(temp)
    return submatrix
