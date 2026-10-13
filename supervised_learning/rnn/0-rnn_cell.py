#!/usr/bin/env python3
"""RNN cell (RNNs task 0)."""

import numpy as np


class RNNCell:
    """Cell of a simple RNN.

    Attributes:
        Wh (np.ndarray): weights for the concatenated hidden state and
            input data, shape (i + h, h), from a normal distribution.
        Wy (np.ndarray): weights for the output, shape (h, o), from a
            normal distribution.
        bh (np.ndarray): bias for the concatenated hidden state and
            input data, shape (1, h), initialized as zeros.
        by (np.ndarray): bias for the output, shape (1, o),
            initialized as zeros.
    """

    def __init__(self, i, h, o):
        """Initialize the RNN cell.

        Args:
            i (int): dimensionality of the data.
            h (int): dimensionality of the hidden state.
            o (int): dimensionality of the outputs.
        """
        self.i = i
        self.h = h
        self.o = o

        # Weight and bias initialization
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (np.ndarray): previous hidden state of shape (m, h).
            x_t (np.ndarray): data input for the cell of shape (m, i).

        Returns:
            tuple: (h_next, y)
                h_next (np.ndarray): the next hidden state of shape
                    (m, h).
                y (np.ndarray): the output of the cell of shape (m, o),
                    using a softmax activation.
        """
        # Concatenate hidden state and input: shape (m, i + h)
        concat = np.concatenate([h_prev, x_t], axis=1)

        # Compute next hidden state: shape (m, h)
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        # Compute output: shape (m, o) with softmax
        # Softmax: exp(x) / sum(exp(x), axis=1, keepdims=True)
        exp_scores = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return (h_next, y)
