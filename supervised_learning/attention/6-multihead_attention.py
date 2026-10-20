#!/usr/bin/env python3
"""Multi head attention (attention task 6)."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi head attention.

    Attributes:
        h (int): the number of heads.
        dm (int): the dimensionality of the model.
        depth (int): the depth of each attention head.
        Wq (tf.keras.layers.Dense): dm units, generates the query matrix.
        Wk (tf.keras.layers.Dense): dm units, generates the key matrix.
        Wv (tf.keras.layers.Dense): dm units, generates the value matrix.
        linear (tf.keras.layers.Dense): dm units, generates the
            attention output.
    """

    def __init__(self, dm, h):
        """Initialize the multi head attention layer.

        Args:
            dm (int): the dimensionality of the model.
            h (int): the number of heads.

        dm is divisible by h.
        """
        pass

    def call(self, Q, K, V, mask):
        """Call the multi head attention layer.

        Args:
            Q: tensor of shape (batch, seq_len_q, dk) containing the
                input to generate the query matrix.
            K: tensor of shape (batch, seq_len_v, dk) containing the
                input to generate the key matrix.
            V: tensor of shape (batch, seq_len_v, dv) containing the
                input to generate the value matrix.
            mask: always None.

        Returns:
            tuple: (output, weights)
                output: tensor whose last two dimensions are
                    (..., seq_len_q, dm) containing the scaled dot
                    product attention.
                weights: tensor whose last three dimensions are
                    (..., h, seq_len_q, seq_len_v) containing the
                    attention weights.
        """
        pass
