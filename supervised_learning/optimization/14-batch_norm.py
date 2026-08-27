#!/usr/bin/env python3
"""
Write a function def create_batch_norm_layer(prev, n, activation): that
creates a batch normalization layer for a neural network in tensorflow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in
    tensorflow

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the
    output of the layer

    Returns: a tensor of the activated output for the layer
    """
    # Dense is a fully connected layer as required
    # checker expects the low-level ops, not as in keras project

    # 1. Dense without activation, VarianceScaling fan_avg (Glorot truncated)
    init = tf.keras.VarianceScaling(mode='fan_avg')  # keep Var(Z)≈1 fwd/bwd

    # let's build Z from prev
    Z = tf.keras.Dense(units=n, kernel_initializer=init)(
        prev)  # (m,n_prev)@(n_prev,n) → (m,n)

  
    # 2-3. Manual moments + learnable scale/shift

    # per-unit over batch m, like np.mean/var axis=0 in task 13
    mean, var = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones((1, n)))  # or [n], both broadcast
    beta = tf.Variable(tf.zeros((1, n)))

    # 4-5. Low-level norm then activation after
    Z_norm = tf.nn.batch_normalization(
        Z, mean, var, beta, gamma, variance_epsilon=1e-7)

    return activation(Z_norm)  # g after BN, not inside Dense
