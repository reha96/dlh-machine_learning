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


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells : List of RNNCell instances of length l.
        X : Data of shape (t, m, i).
        h_0 : Initial hidden states of shape (l, m, h).

    Returns:
        tuple: (H, Y)
            H is a numpy.ndarray of shape (t + 1, l, m, h) containing
              all hidden states for every layer and time step.
            Y is a numpy.ndarray of shape (t, m, o) containing
              all outputs from the final layer.
    """
    lrc = len(rnn_cells)
    t, m, i = X.shape
    h = h_0.shape[2]
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, lrc, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x_t = X[step]

        for layer in range(lrc):
            h_prev = H[step, layer]

            if layer == 0:
                x_input = x_t
            else:
                x_input = H[step + 1, layer - 1]

            h_next, y = rnn_cells[layer].forward(h_prev, x_input)

            H[step + 1, layer] = h_next

            if layer == lrc - 1:
                Y[step] = y

    return H, Y
