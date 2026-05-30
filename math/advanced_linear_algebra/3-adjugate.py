#!/usr/bin/env python3
"""Write a function def adjugate(matrix): that calculates
    the adjugate matrix of a matrix:
    """


def determinant(matrix):
    """calculates
    the determinant of a matrix

    Args:
        matrix (_type_): _description_

    Raises:
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
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
        return matrix[0][0]

    # Base case: 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    submatrix = []
    for col in range(len(matrix)):  # col = 0, 1, 2
        temp = []
        for row_idx in range(1, len(matrix)):  # skip first row (index 0)
            # remove the current column from this row
            new_row = matrix[row_idx][:col] + matrix[row_idx][col+1:]
            temp.append(new_row)
        submatrix.append(temp)

    det = 0
    for col in range(len(matrix)):
        det += ((-1) ** col) * matrix[0][col] * determinant(submatrix[col])
    return det


def minor(matrix):
    """calculates the minor
matrix of a matrix

    Args:
        matrix (_type_): _description_
    """

    if not isinstance(matrix, list) \
            or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not all(len(row) == len(matrix) for row in matrix) \
            or matrix == [] or matrix == [[]] or len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Base case: 1x1 matrix
    if len(matrix) == 1:
        return [[1]]

    # Base case: 2x2 matrix
    out = []
    if len(matrix) == 2:
        for row in range(len(matrix)-1, -1, -1):
            out.append(matrix[row][::-1])
        return out

    # Base case: 3x3 matrix and more
    out = []
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
        temp.append(determinant(mat))
    for i in range(len(matrix)):  # nice way to create a list of list
        out += [temp[i*len(matrix):(i+1)*len(matrix)]]

    return out


def cofactor(matrix):
    """calculates
the cofactor matrix of a matrix

    Args:
        matrix (_type_): _description_
    """
    min = minor(matrix)

    temp = []
    for row in range(len(min)):
        for col in range(len(min)):
            temp.append((-1) ** (col+row) * min[row][col])
    out = []
    for i in range(len(min)):
        out += [temp[i*len(min):(i+1)*len(min)]]
    return out


def adjugate(matrix):
    """calculates the adjugate matrix of a matrix by
    transposing the cofactor matrix 

    Args:
        matrix (_type_): _description_
    """

    cofact = cofactor(matrix)

    temp = []
    for row in range(len(cofact)):
        for col in range(len(cofact)):
            temp.append(cofact[col][row])

    out = []
    for i in range(len(matrix)):  # nice way to create a list of list
        out += [temp[i*len(matrix):(i+1)*len(matrix)]]

    return out
