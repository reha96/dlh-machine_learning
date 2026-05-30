#!/usr/bin/env python3
"""Write a function def definiteness(matrix):
    that calculates the definiteness of a matrix:
    """
import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix
    using eigenvals from numpy

    Args:
        matrix (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    trans = np.transpose(matrix)

    if trans != matrix:
        return None

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if not all(np.shape(matrix)[0] == i for i in np.shape(matrix)):
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
