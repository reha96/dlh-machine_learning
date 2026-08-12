#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: the labels to convert
        classes (int): the number of classes

    Returns:
        the one-hot matrix
    """
    # one row per label, a single 1 marks the label's class (wide to long)
    return K.utils.to_categorical(labels, classes)
