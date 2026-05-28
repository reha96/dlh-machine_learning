#!/usr/bin/env python3

"""Write a function def determinant(matrix): that calculates
    the determinant of a matrix:
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
    if not isinstance(matrix, list) or not isinstance(matrix[0], list) \
            or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Base case: 0x0 matrix
    if matrix == [[]]:
        return 1

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
