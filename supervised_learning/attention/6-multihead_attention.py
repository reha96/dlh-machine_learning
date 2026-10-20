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
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

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
        batch = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split the model depth across the h heads
        Q = self._split(batch, Q)
        K = self._split(batch, K)
        V = self._split(batch, V)

        scaled, weights = sdp_attention(Q, K, V, mask)

        # Merge the heads back into the model depth
        scaled = tf.transpose(scaled, perm=[0, 2, 1, 3])
        merged = tf.reshape(scaled, (batch, -1, self.dm))

        return self.linear(merged), weights

    def _split(self, batch, x):
        """Reshape (batch, seq, dm) into (batch, h, seq, depth)."""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
