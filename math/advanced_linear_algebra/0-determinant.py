#!usr/bin/env python3
import numpy as np

"""Write a function def determinant(matrix): that calculates
    the determinant of a matrix:
    """
matrix3 = [[1,2,3],[4,5,6],[7,8,9]]
matrix2 = [[1,2],[4,5]]


def determinant(matrix):
    if not isinstance(matrix, list) and not isinstance(matrix[0], list):
        raise TypeError
    if len(matrix) != len(matrix[0]):
        raise ValueError
    
    row = matrix[0]
    # write case for 1x1
    if len(row) == 1:
        return(matrix[0])
    # write case for 2x2
    if len(row) == 2:
        first = matrix[0][0] * matrix[1][1]
        second = matrix[1][0] * matrix[0][1]
        det = first - second
        return(det)
    # write case for 3x3    
    if len(row) == 3:
        det_mats = []
        for col in range(len(matrix)):  # col = 0, 1, 2
            submat = []
            for row_idx in range(1, len(matrix)): # skip first row (index 0)
                # remove the current column from this row
                new_row = matrix[row_idx][:col] + matrix[row_idx][col+1:]
                submat.append(new_row)
            det_mats.append(submat)    
        
        det = 0
        for i in range(len(row)):
            if i % 2 == 0:
                det += row[i] * (det_mats[i][0][0] * det_mats[i][1][1] - det_mats[i][1][0] * det_mats[i][0][1])
            else:
                det -= row[i] * (det_mats[i][0][0] * det_mats[i][1][1] - det_mats[i][1][0] * det_mats[i][0][1])
        print(det) 
    
        # a = matrix[0][0]
        # first = matrix[1][1] * matrix[2][2]
        # second = matrix[2][1] * matrix[1][2]
        # det1 = first - second
        # b = matrix[0][1]
        # first = matrix[1][0] * matrix[2][2]
        # second = matrix[2][0] * matrix[1][2]
        # det2 = first - second
        # c = matrix[0][2]
        # first = matrix[1][0] * matrix[2][1]
        # second = matrix[2][0] * matrix[1][1]
        # det3 = first - second
        # det = a*det1 - b*det2 + c*det3
        # print(det) 
    # write case for 4x4
    
determinant(matrix3)
print(np.linalg.det(matrix3))
