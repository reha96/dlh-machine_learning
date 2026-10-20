#!/usr/bin/env python3
"""Transformer decoder block (attention task 8)."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Create a decoder block for a transformer.

    Attributes:
        mha1 (MultiHeadAttention): the first multi head attention layer.
        mha2 (MultiHeadAttention): the second multi head attention layer.
        dense_hidden (tf.keras.layers.Dense): hidden units, relu.
        dense_output (tf.keras.layers.Dense): dm units.
        layernorm1 (tf.keras.layers.LayerNormalization): epsilon=1e-6.
        layernorm2 (tf.keras.layers.LayerNormalization): epsilon=1e-6.
        layernorm3 (tf.keras.layers.LayerNormalization): epsilon=1e-6.
        dropout1 (tf.keras.layers.Dropout): the first dropout layer.
        dropout2 (tf.keras.layers.Dropout): the second dropout layer.
        dropout3 (tf.keras.layers.Dropout): the third dropout layer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the decoder block.

        Args:
            dm (int): the dimensionality of the model.
            h (int): the number of heads.
            hidden (int): the number of hidden units in the fully
                connected layer.
            drop_rate (float): the dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Call the decoder block.

        Args:
            x: tensor of shape (batch, target_seq_len, dm) containing
                the input to the decoder block.
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder.
            training (bool): whether the model is training.
            look_ahead_mask: the mask to be applied to the first multi
                head attention layer.
            padding_mask: the mask to be applied to the second multi
                head attention layer.

        Returns:
            tensor of shape (batch, target_seq_len, dm) containing the
                block's output.
        """
        # Masked self attention over the target, residual, layer norm
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        x_norm1 = self.layernorm1(x + self.dropout1(attn1, training=training))

        # Cross attention over the encoder output, residual, layer norm
        attn2, _ = self.mha2(x_norm1, encoder_output, encoder_output,
                             padding_mask)
        x_norm2 = self.layernorm2(
            x_norm1 + self.dropout2(attn2, training=training))

        # Position wise feed forward, residual, layer norm
        hidden = self.dense_hidden(x_norm2)
        out = self.dense_output(hidden)

        return self.layernorm3(
            x_norm2 + self.dropout3(out, training=training))
