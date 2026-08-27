#!/usr/bin/env python3
"""
Write a function def create_momentum_op(alpha, beta1): that sets up the
gradient descent with momentum optimization algorithm in TensorFlow
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in
    TensorFlow

    alpha is the learning rate
    beta1 is the momentum weight

    Returns: optimizer
    """
    # all we did previously, with TensorFlow
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
