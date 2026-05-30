#!/usr/bin/env python3
"""Write a function def definiteness(matrix):
    that calculates the definiteness of a matrix:
    """
import numpy as np


def definiteness(matrix):

    if not isinstance(matrix, list) \
            or not all(isinstance(row, list) for row in matrix):
        return None

    if not all(len(row) == len(matrix) for row in matrix) \
            or matrix == [] or matrix == [[]] or len(matrix) == 0:
        return None

    tol = 1e-10
    eigvals = np.linalg.eigvalsh(matrix)  # works for symmetric matrices
    if np.all(eigvals > tol):
        return "Positive definite"
    elif np.all(eigvals >= -tol):     # allows near-zero eigenvalues
        return "Positive semi-definite"
    elif np.all(eigvals < -tol):
        return "Negative definite"
    elif np.all(eigvals <= tol):
        return "Negative semi-definite"
    else:
        return "Indefinite"
