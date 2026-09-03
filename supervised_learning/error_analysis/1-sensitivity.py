#!/usr/bin/env python3
"""Calculates sensitivity for each class in a confusion matrix."""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity
        of each class.
    """
    pass
