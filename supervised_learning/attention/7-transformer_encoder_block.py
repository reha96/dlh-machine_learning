#!/usr/bin/env python3
"""Transformer encoder block (attention task 7)."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Create an encoder block for a transformer.

    Attributes:
        mha (MultiHeadAttention): a multi head attention layer.
        dense_hidden (tf.keras.layers.Dense): hidden units, relu.
        dense_output (tf.keras.layers.Dense): dm units.
        layernorm1 (tf.keras.layers.LayerNormalization): epsilon=1e-6.
        layernorm2 (tf.keras.layers.LayerNormalization): epsilon=1e-6.
        dropout1 (tf.keras.layers.Dropout): the first dropout layer.
        dropout2 (tf.keras.layers.Dropout): the second dropout layer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the encoder block.

        Args:
            dm (int): the dimensionality of the model.
            h (int): the number of heads.
            hidden (int): the number of hidden units in the fully
                connected layer.
            drop_rate (float): the dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Call the encoder block.

        Args:
            x: tensor of shape (batch, input_seq_len, dm) containing
                the input to the encoder block.
            training (bool): whether the model is training.
            mask: the mask to be applied for multi head attention.

        Returns:
            tensor of shape (batch, input_seq_len, dm) containing the
                block's output.
        """
        # Self attention, then residual and layer norm
        attn, _ = self.mha(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn, training=training))

        # Position wise feed forward, then residual and layer norm
        hidden = self.dense_hidden(x)
        out = self.dense_output(hidden)

        return self.layernorm2(x + self.dropout2(out, training=training))
