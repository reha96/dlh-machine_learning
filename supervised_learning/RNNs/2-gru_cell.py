#!/usr/bin/env python3
# (Based on 1-rnn.py)
"""Gated Recurrent Unit cell (RNNs task 2)."""

import numpy as np


class GRUCell:
    """Cell of a Gated Recurrent Unit.

    Attributes:
        Wz (np.ndarray): weights for the update gate, shape
            (i + h, h), from a normal distribution.
        Wr (np.ndarray): weights for the reset gate, shape
            (i + h, h), from a normal distribution.
        Wh (np.ndarray): weights for the candidate hidden state, shape
            (i + h, h), from a normal distribution.
        Wy (np.ndarray): weights for the output, shape (h, o), from a
            normal distribution.
        bz (np.ndarray): bias for the update gate, shape (1, h),
            initialized as zeros.
        br (np.ndarray): bias for the reset gate, shape (1, h),
            initialized as zeros.
        bh (np.ndarray): bias for the candidate hidden state, shape
            (1, h), initialized as zeros.
        by (np.ndarray): bias for the output, shape (1, o),
            initialized as zeros.
    """

    def __init__(self, i, h, o):
        """Initialize the GRU cell.

        Args:
            i (int): dimensionality of the data.
            h (int): dimensionality of the hidden state.
            o (int): dimensionality of the outputs.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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
        concat = np.concatenate([h_prev, x_t], axis=1)

        # Update gate (sigmoid)
        z = 1 / (1 + np.exp(-(np.dot(concat, self.Wz) + self.bz)))

        # Reset gate (sigmoid)
        r = 1 / (1 + np.exp(-(np.dot(concat, self.Wr) + self.br)))

        # Candidate hidden state
        r_h_prev = r * h_prev
        concat_r = np.concatenate([r_h_prev, x_t], axis=1)
        h_tilde = np.tanh(np.dot(concat_r, self.Wh) + self.bh)

        # Gated hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output with softmax
        exp_scores = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return (h_next, y)
