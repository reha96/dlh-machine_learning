#!/usr/bin/env python3
"""
Write a function def create_RMSProp_op(alpha, beta2, epsilon): that sets
up the RMSProp optimization algorithm in TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow

    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero

    Returns: optimizer
    """
    return tf.keras.optimizers.RMSprop(earning_rate=alpha,
                                       rho=beta2,
                                       epsilon=epsilon)
