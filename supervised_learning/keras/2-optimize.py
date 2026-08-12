#!/usr/bin/env python3
"""Sets up Adam optimization for a Keras model with categorical
crossentropy loss."""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical crossentropy
    loss and accuracy metrics.

    Args:
        network: the model to optimize
        alpha (float): the learning rate
        beta1 (float): the first Adam optimization parameter
        beta2 (float): the second Adam optimization parameter

    Returns:
        None
    """
    pass
