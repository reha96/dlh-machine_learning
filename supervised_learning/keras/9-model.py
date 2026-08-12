#!/usr/bin/env python3
"""Saves and loads a Keras model."""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    Args:
        network: the model to save
        filename (str): the path of the file that the model should be saved
            to

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model.

    Args:
        filename (str): the path of the file that the model should be loaded
            from

    Returns:
        the loaded model
    """
    K.models.load_model(filename)
