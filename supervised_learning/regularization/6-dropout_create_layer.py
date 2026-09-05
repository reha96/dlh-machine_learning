#!/usr/bin/env python3
"""Dense layer with Dropout (TensorFlow)."""
# Provisional template: signature confirmed across both reference forks;
# pending intranet project 2297 spec check after sign-in. No solution code.
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    prev is a tensor containing the output of the previous layer.
    n is the number of nodes the new layer should contain.
    activation is the activation function to use on the layer.
    keep_prob is the probability that a node will be kept.
    training is a boolean indicating whether the model is in training mode.

    Returns: the output of the new layer.
    """
    pass
