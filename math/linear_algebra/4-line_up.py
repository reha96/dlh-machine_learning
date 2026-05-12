#!/usr/bin/env python3
"""Write a function def add_arrays(arr1, arr2): that adds two arrays
element-wise:

    You can assume that arr1 and arr2 are lists of ints/floats
    You must return a new list
    If arr1 and arr2 are not the same shape, return None

    """


def add_arrays(arr1, arr2):
    """function def add_arrays(arr1, arr2): that adds two arrays
element-wise

    Args:
        arr1 (_type_): _description_
        arr2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(arr1) != len(arr2):
        return None
    out = []
    for i in range(len(arr1)):
        out.append(arr1[i]+arr2[i])
    return out
