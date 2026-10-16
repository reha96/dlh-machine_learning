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
    if not isinstance(X, np.ndarray) or not isinstance(h_0, np.ndarray):
        raise TypeError("X and h_0 must be numpy.ndarray")

    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.by.o

    if h_0.ndim != 2 or h_0.shape[0] != m:
        raise ValueError("h_0 must have shape (m, h) compatible with X")

    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y

    return (H, Y)
