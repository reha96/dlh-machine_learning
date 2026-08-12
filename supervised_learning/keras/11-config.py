#!/usr/bin/env python3
"""Saves and loads a model's configuration in JSON format."""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Args:
        network: the model whose configuration should be saved
        filename (str): the path of the file that the configuration should
            be saved to

    Returns:
        None
    """
    network.to_json(filename)


def load_config(filename):
    """
    Loads a model with a specific configuration.

    Args:
        filename (str): the path of the file containing the model's
            configuration in JSON format

    Returns:
        the loaded model
    """
    K.models.model_from_json(filename)
