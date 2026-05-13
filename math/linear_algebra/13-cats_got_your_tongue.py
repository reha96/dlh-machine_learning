#!/usr/bin/env python3
import numpy as np


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
