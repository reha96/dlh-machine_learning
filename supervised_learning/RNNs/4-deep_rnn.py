#!/usr/bin/env python3
# (Based on 3-lstm_cell.py)
"""Deep RNN forward propagation (RNNs task 4)."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells (list): list of cell instances, one per layer.
        X (np.ndarray): data of shape (t, m, i) — t time steps, batch
            size m, input dimensionality i.
        h_0 (np.ndarray): initial hidden states of shape (l, m, h) —
            one initial state per layer.

    Returns:
        tuple: (H, Y)
            H (np.ndarray): hidden states of every layer for every
                time step of shape (t + 1, l, m, h), including the
                initial hidden states at index 0.
            Y (np.ndarray): outputs of the last layer for every time
                step of shape (t, m, o).

    Raises:
        TypeError: if X or h_0 is not a `numpy.ndarray`.
        ValueError: if X and h_0 have incompatible shapes.
    """
    pass
