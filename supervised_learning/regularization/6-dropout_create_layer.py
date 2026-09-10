#!/usr/bin/env python3
"""Dense layer with Dropout (TensorFlow)."""
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
    # layer weights
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # create layer and store output
    # activation inside Dense, dropout comes after
    dense_out = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init)(prev)

    # add dropout
    rate = 1-keep_prob  # dropout rate
    return tf.keras.layers.Dropout(rate=rate)(dense_out, training=training)
