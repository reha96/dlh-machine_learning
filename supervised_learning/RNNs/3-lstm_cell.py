#!/usr/bin/env python3
# (Based on 2-gru_cell.py)
"""Long Short-Term Memory cell (RNNs task 3)."""

import numpy as np


class LSTMCell:
    """Cell of a Long Short-Term Memory unit.

    Attributes:
        Wf (np.ndarray): weights for the forget gate, shape
            (i + h, h), from a normal distribution.
        Wu (np.ndarray): weights for the update gate, shape
            (i + h, h), from a normal distribution.
        Wc (np.ndarray): weights for the intermediate cell state,
            shape (i + h, h), from a normal distribution.
        Wo (np.ndarray): weights for the output gate, shape
            (i + h, h), from a normal distribution.
        Wy (np.ndarray): weights for the output, shape (h, o), from a
            normal distribution.
        bf (np.ndarray): bias for the forget gate, shape (1, h),
            initialized as zeros.
        bu (np.ndarray): bias for the update gate, shape (1, h),
            initialized as zeros.
        bc (np.ndarray): bias for the intermediate cell state, shape
            (1, h), initialized as zeros.
        bo (np.ndarray): bias for the output gate, shape (1, h),
            initialized as zeros.
        by (np.ndarray): bias for the output, shape (1, o),
            initialized as zeros.
    """

    def __init__(self, i, h, o):
        """Initialize the LSTM cell.

        Args:
            i (int): dimensionality of the data.
            h (int): dimensionality of the hidden state.
            o (int): dimensionality of the outputs.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (np.ndarray): previous hidden state of shape (m, h).
            c_prev (np.ndarray): previous cell state of shape (m, h).
            x_t (np.ndarray): data input for the cell of shape (m, i).

        Returns:
            tuple: (h_next, c_next, y)
                h_next (np.ndarray): the next hidden state of shape
                    (m, h).
                c_next (np.ndarray): the next cell state of shape
                    (m, h).
                y (np.ndarray): the output of the cell of shape (m, o),
                    using a softmax activation.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate (sigmoid)
        f = np.matmul(concat, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))

        # Update gate (sigmoid)
        u = np.matmul(concat, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))

        # Candidate cell state
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f * c_prev + u * c_tilde

        # Output gate (sigmoid)
        o = np.matmul(concat, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output with softmax
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)

        return (h_next, c_next, y)
