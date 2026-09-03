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
    # TN is 
