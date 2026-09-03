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
    # get all previous calculations
    true_p = np.diag(confusion)

    # sum of non diag elements ROW-WISE (across wrong predictions) equal FN
    row_sum = np.sum(confusion, axis=1)
    false_n = row_sum - true_p

    # sum of non diag elements COL-WISE (across true labels) equal FP
    col_sum = np.sum(confusion, axis=0)
    false_p = col_sum - true_p

    # according to formula
    true_n = np.sum(confusion) - true_p - false_p - false_n

    # f1 is 2 * (precision * recall) / (precision + recall)
    # simplifies to TP / (TP + (FN+FP)/2)
    f1 = true_p / (true_p + (false_n+false_p)/2)
    return f1
