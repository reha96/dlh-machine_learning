#!/usr/bin/env python3
"""Calculates F1 score for each class in a confusion matrix."""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score
        of each class.
    """
    pass
