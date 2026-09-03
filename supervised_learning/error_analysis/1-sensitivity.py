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
    # sensitivity is Recall = True Positives (TP) / (TP + False Negatives (FN))
    # diagonals store TPs per class
    true_p = np.diag(confusion)

    # sum of non diag elements ROW-WISE (across wrong predictions) equal FN
    row_sum = np.sum(confusion, axis=1)
    false_p = row_sum - true_p

    return true_p/(true_p+false_p)


if __name__ == '__main__':
    confusion = np.load(
        '/home/rehat/Documents/GitHub/dlh-machine_learning/supervised_learning/error_analysis/confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(sensitivity(confusion))
