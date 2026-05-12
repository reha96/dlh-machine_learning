#!/usr/bin/env python3
"""Write a function def cat_arrays(arr1, arr2): that concatenates
two arrays:

    You can assume that arr1 and arr2 are lists of ints/floats
    You must return a new list

    """


def cat_arrays(arr1, arr2):
    """function def cat_arrays(arr1, arr2): that concatenates
two arrays

    Args:
        arr1 (_type_): _description_
        arr2 (_type_): _description_
    """
    out = []
    out.extend(arr1)
    out.extend(arr2)
    return out
