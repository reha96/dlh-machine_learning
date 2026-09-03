#!/usr/bin/env python3
"""Calculates specificity for each class in a confusion matrix."""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels.

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
        of each class.
    """

    # specificity is 1 - True Negative Rate (over total negatives)
    # which is True Negatives (TN) / (TN + False Positives (FP))
    # TN = Total Samples − (TP+FP+FN)

    # get all previous calculations
    true_p = np.diag(confusion)

    # sum of non diag elements ROW-WISE (across wrong predictions) equal FN
    row_sum = np.sum(confusion, axis=1)
    false_n = row_sum - true_p

    # sum of non diag elements COL-WISE (across true labels) equal FP
    col_sum = np.sum(confusion, axis=0)
    false_p = col_sum - true_p

    true_n = np.sum(confusion) - true_p - false_p - false_n
    return true_n / (true_n + false_p)
