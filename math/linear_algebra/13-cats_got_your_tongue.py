#!/usr/bin/env python3
import numpy as np

# """Write a function def np_cat(mat1, mat2, axis=0)
# that concatenates two matrices along a specific axis:

#     You can assume that mat1 and mat2 can be interpreted
#     as numpy.ndarrays
#     You must return a new numpy.ndarray
#     You are not allowed to use any loops or conditional
#     statements
#     You may use: import numpy as np
#     You can assume that mat1 and mat2 are never empty

#     """


def np_cat(mat1, mat2, axis=0):
    """a function def np_cat(mat1, mat2, axis=0)
    that concatenates two matrices along a specific axis

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.
    """
    return np.concatenate((mat1, mat2),
                          axis)
