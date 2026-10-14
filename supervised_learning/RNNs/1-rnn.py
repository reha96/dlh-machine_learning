#!/usr/bin/env python3
"""Simple RNN forward propagation (RNNs task 1)."""
# (Based on 0-rnn_cell.py)
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
    # raises
    if not isinstance(X, np.ndarray) or not isinstance(h_0, np.ndarray):
        raise TypeError("X or h_0 is not a `numpy.ndarray`")

    if X.shape[1] != h_0.shape[0]:
        raise ValueError("X and h_0 have incompatible shapes")

    # unpack to get dimensions
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.o

    # initiate
    H = np.zeros(t + 1, m, h)
    Y = np.zeros(t, m, o)

    # loop
    for 
    
    return (H, Y)
