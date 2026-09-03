#!/usr/bin/env python3
"""Calculates precision for each class in a confusion matrix."""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) containing the precision
        of each class.
    """
    # precision is True Positives (TP) / (TP + False Positives (FP))
    # diagonals store TPs per class
    true_p = np.diag(confusion)

    # sum of non diag elements COL-WISE (across true labels) equal FP
    col_sum = np.sum(confusion, axis=0)
    false_p = col_sum - true_p

    return true_p/(true_p+false_p)
