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

    # Base case: 0x0 matrix
    if matrix == [[]]:
        return 1

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Every element must be a list
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if len(matrix) == 1:
        return 1

    # Base case: 2x2 matrix
    out = []
    if len(matrix) == 2:
        for row in range(len(matrix)-1, -1, -1):
            out.append(matrix[row][::-1])
        return out

    # Base case: 3x3 matrix and more
    submatrix = []
    for i in range(len(matrix)):  # row = 0, 1, 2, ...
        for j in range(len(matrix)):  # col = 0, 1, 2, ...
            temp = []
            for k in range(len(matrix)):
                if k == i:
                    continue
                new_row = matrix[k][:j] + matrix[k][j+1:]
                temp.append(new_row)
            submatrix.append(temp)

    temp = []
    for mat in submatrix:
        temp.append(mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0])
    for i in range(len(matrix)):
        out += [temp[i*len(matrix):(i+1)*len(matrix)]]

    return out
