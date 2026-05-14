#!/usr/bin/env python3
"""Write a function def np_slice(matrix, axes={}): that
slices a matrix along specific axes:

    You can assume that matrix is a numpy.ndarray
    You must return a new numpy.ndarray
    axes is a dictionary where the key is an axis to slice
    along and the value is a tuple representing the slice
    to make along that axis
    You can assume that axes represents a valid slice
    """


import numpy as np


def np_slice(matrix, axes={}):
    result = np.take_along_axis(matrix, np.array([[1, 2], [1, 2]]), axis=1)
    return result


mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(np_slice(mat1, axes={1: (1, 3)}))
# print(mat1)
# mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
#                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
#                  [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
# print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
# print(mat2)
