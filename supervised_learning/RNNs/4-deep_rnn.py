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
    if not isinstance(X, np.ndarray) or not isinstance(h_0, np.ndarray):
        raise TypeError("X and h_0 must be numpy.ndarray")

    t, m, i = X.shape
    n_layers = h_0.shape[0]
    h = h_0.shape[2]

    if len(rnn_cells) != n_layers:
        raise ValueError("Number of rnn_cells must match "
                         "number of layers in h_0")

    H = np.zeros((t + 1, n_layers, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, rnn_cells[-1].o))

    for step in range(t):
        # Propulate through all layers at this time step.
        # Layer 0 takes the original input X[step] with previous state h_0[0].
        h_prev = rnn_cells[0].forward(h_0[0], X[step])[0]
        H[step + 1, 0] = h_prev
        # Remaining layers take the previous layer's hidden state as h_prev,
        # and the original input X[step] as x_t.
        for layer_idx in range(1, n_layers):
            h_next, _ = rnn_cells[layer_idx].forward(h_prev, X[step])
            h_prev = h_next
            H[step + 1, layer_idx] = h_next

    # Compute outputs from the last layer at each time step.
    for step in range(t):
        # Re-propulate through layers to get the last layer's hidden state.
        h_prev = h_0[0].copy()
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                h_next, _ = rnn_cells[layer_idx].forward(
                    h_0[layer_idx], X[step])
            else:
                h_next, _ = rnn_cells[layer_idx].forward(h_prev, X[step])
            h_prev = h_next
        # Softmax output from the last layer.
        exp_scores = np.dot(h_prev, rnn_cells[-1].Wy) + rnn_cells[-1].by
        probs = np.exp(exp_scores)
        Y[step] = probs / np.sum(probs, axis=1, keepdims=True)

    return (H, Y)
