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
        pass

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
        pass
