#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    Args:
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        W (numpy.ndarray): of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution.
        b (numpy.ndarray): of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution.
        activation: activation function applied to the convolution.
        padding (str): either 'same' or 'valid', indicating the type
            of padding used.
        stride (tuple): (sh, sw) containing the strides for the convolution.

    Returns:
        numpy.ndarray: the output of the convolutional layer.
    """
    # extract relevant shapes
    # input (A) obs, height, width, channels
    m, h_prev, w_prev, c_prev = A_prev.shape
    # kernel (W) height, width, channels
    kh, kw, c_prev, c_new = W.shape
    # step size height, width
    s_h, s_w = stride

    # add padding for input (A) kernel area
    if padding == "valid":
        ph, pw = 0, 0
    else:
        # padding formula for height, width
        ph = ((h_prev-1)*s_h + kh - h_prev)//2
        pw = ((w_prev-1)*s_w + kw - w_prev)//2

    # padding applied to only height and width (not m, not c)
    padding_dims = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_padded = np.pad(A_prev, padding_dims, mode="constant")

    # calculate the output dimensions
    h_new = (h_prev + 2*ph - kh)//s_h + 1
    w_new = (w_prev + 2*pw - kw)//s_w + 1
    convolved = np.zeros((m, h_new, w_new, c_new))

    # sweep each filter across the padded input
    for i in range(h_new):
        for j in range(w_new):
            # region of the input covered by the filter
            h_start = i*s_h
            w_start = j*s_w
            region = A_padded[:, h_start:h_start+kh, w_start:w_start+kw, :]

            # dot product of the region with each filter
            for k in range(c_new):
                kernel = W[:, :, :, k]
                convolved[:, i, j, k] = np.sum(region*kernel, axis=(1, 2, 3))

    # add bias and apply activation
    Z = convolved + b
    return activation(Z)
