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
    
    try:
        if axis == 0:
            out = []
            out.extend([mat1, mat2])
        if axis == 1:
            # rows =
            # columns =
            # out = [[0] * len(mat1[0]) for _ in range(len(mat1))]
            # for i in range(len(mat1)):
            #         out[i][j] = (mat1[i][j] + mat2[i][j])
    except Exception:
        return None
    return out


mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)
print(mat5)
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 5], [3, 4, 6]]
# [[1, 2, 7], [3, 4, 8]]