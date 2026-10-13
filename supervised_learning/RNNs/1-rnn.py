#!/usr/bin/env python3
# (Based on 0-rnn_cell.py)
"""Simple RNN forward propagation (RNNs task 1)."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell (RNNCell): an instance of the `RNNCell` class to use
            at every time step.
        X (np.ndarray): data of shape (t, m, i) — t time steps, batch
            size m, input dimensionality i.
        h_0 (np.ndarray): initial hidden state of shape (m, h).

    Returns:
        tuple: (H, Y)
            H (np.ndarray): hidden states for every time step of shape
                (t + 1, m, h), including the initial hidden state at
                index 0.
            Y (np.ndarray): outputs for every time step of shape
                (t, m, o).

    Raises:
        TypeError: if X or h_0 is not a `numpy.ndarray`.
        ValueError: if X and h_0 have incompatible shapes.
    """
    pass
