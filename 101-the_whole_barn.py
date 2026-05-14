#!/usr/bin/env python3
"""Write a function def add_matrices(mat1, mat2):
that adds two matrices:

    You can assume that mat1 and mat2 are matrices
    containing ints/floats
    You can assume all elements in the same
    dimension are of the same type/shape
    You must return a new matrix
    If matrices are not the same shape, return None
    You can assume that mat1 and mat2 will never be empty

    """


def add_matrices(mat1, mat2):
    """function def add_matrices(mat1, mat2):
that adds two matrices

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Check matrix dimensions
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None  # Mismatched top-level lengths

        result = []
        for i in range(len(mat1)):
            # Call add_matrices recursively for each pair of elements
            added = add_matrices(mat1[i], mat2[i])
            if added is None:  # If any inner call failed
                return None
            result.append(added)

        return result
    else:
        # Base case: both mat1 and mat2 must be numbers
        return mat1 + mat2
