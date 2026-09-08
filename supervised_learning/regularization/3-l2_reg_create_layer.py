#!/usr/bin/env python3
"""Dense layer with L2 regularization (TensorFlow)."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization.

    prev is a tensor containing the output of the previous layer.
    n is the number of nodes the new layer should contain.
    activation is the activation function to use on the layer.
    lambtha is the L2 regularization parameter.

    Returns: the output of the new layer.
    """
    # He et al. initialization
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    # L2 penalty for this layer
    reg = tf.keras.regularizers.L2(lambtha)
    # Dense layer with L2 on kernel only (biases not regularized)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    return layer(prev)
