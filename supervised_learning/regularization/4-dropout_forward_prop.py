#!/usr/bin/env python3
"""Forward propagation with Dropout (numpy)."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    X is a numpy.ndarray of shape (nx, m) containing the input data.
    weights is a dictionary of the weights and biases of the neural
    network.
    L is the number of layers in the network.
    keep_prob is the probability that a node will be kept.

    Returns: a dictionary containing the outputs of each layer and the
    dropout mask used on each layer.
    """
    # store the input for backprop later
    cache = {'A0': X}
    A = X

    # layer 1, 2, ..., L
    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        # linear step: every layer, always
        Z = np.matmul(W, A) + b

        if layer < L:
            # HIDDEN layer: tanh, then dropout
            A = np.tanh(Z)

            # (1) coin flip for every neuron of every example
            mask = (np.random.rand(
                A.shape[0], A.shape[1]) < keep_prob).astype(int)

            # (2) dropped neurons output 0
            A *= mask

            # (3) rescale survivors (inverted dropout)
            A /= keep_prob

            # (4) REMEMBER the mask — backprop will need it!
            cache['D' + str(layer)] = mask
        else:
            # OUTPUT layer: softmax, NO dropout
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

        cache['A' + str(layer)] = A        # save every layer's output

    return cache
