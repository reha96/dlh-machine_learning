#!/usr/bin/env python3


def matrix_shape(mat):
    out = []
    while (
        isinstance(mat, list) and len(mat) > 0
    ):  # Loop as long as we have non-empty lists
        out.append(len(mat))  # Append the current dimension
        mat = mat[0]  # Drill down to the next level
    return out  # Return the shape list
