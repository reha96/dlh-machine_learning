#!/usr/bin/env python3
"""Scaled dot product attention (attention task 5)."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculate the scaled dot product attention.

    Args:
        Q: tensor whose last two dimensions are (..., seq_len_q, dk)
            containing the query matrix.
        K: tensor whose last two dimensions are (..., seq_len_v, dk)
            containing the key matrix.
        V: tensor whose last two dimensions are (..., seq_len_v, dv)
            containing the value matrix.
        mask: tensor that can be broadcast into (..., seq_len_q,
            seq_len_v) containing the optional mask, or defaulted to
            None.

    Returns:
        tuple: (output, weights)
            output: tensor whose last two dimensions are
                (..., seq_len_q, dv) containing the scaled dot product
                attention.
            weights: tensor whose last two dimensions are
                (..., seq_len_q, seq_len_v) containing the attention
                weights.
    """
    # (..., seq_len_q, seq_len_v): every query against every key
    logits = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    logits = logits / tf.math.sqrt(dk)

    if mask is not None:
        logits += mask * -1e9

    weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
