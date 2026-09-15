#!/usr/bin/env python3
"""Backward propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a neural
    network.

    Args:
        dZ (numpy.ndarray): of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output
            of the convolutional layer.
            - m is the number of examples
            - h_new is the height of the output
            - w_new is the width of the output
            - c_new is the number of channels in the output
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
        W (numpy.ndarray): of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution.
            - kh is the filter height
            - kw is the filter width
        b (numpy.ndarray): of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution.
        padding (str): either 'same' or 'valid', indicating the type of
            padding used.
        stride (tuple): (sh, sw) containing the strides for the convolution.
            - sh is the stride for the height
            - sw is the stride for the width

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: the partial derivatives
        with respect to the previous layer (dA_prev), the kernels (dW), and
        the biases (db), respectively.
    """
    # extract relevant shapes
    # partial derivatives (dZ)
    m, h_new, w_new, c_new = dZ.shape
    # input (A) obs, height, width, channels
    m, h_prev, w_prev, c_prev = A_prev.shape
    # kernel (W) height, width, channels
    kh, kw, c_prev, c_new = W.shape
    # step size height, width
    sh, sw = stride

    # add padding for input (A) kernel area
    if padding == "valid":
        ph, pw = 0, 0
    else:
        # padding formula for height, width
        ph = ((h_prev-1)*sh + kh - h_prev)//2 + 1
        pw = ((w_prev-1)*sw + kw - w_prev)//2 + 1

    # padding applied to only height and width (not m, not c)
    padding_dims = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_padded = np.pad(A_prev, padding_dims, mode="constant")

    # init the output dimensions
    dA_prev = np.zeros((m, h_prev+2*ph, w_prev+2*pw, c_prev))
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # sweep each output position and scatter the gradients back
    for i in range(h_new):
        for j in range(w_new):
            # region of the padded input covered by the filter
            h_start = i*sh
            w_start = j*sw
            region = A_padded[:, h_start:h_start+kh, w_start:w_start+kw, :]

            # scatter and gather gradients for each filter
            for k in range(c_new):
                dA_prev[:, h_start:h_start+kh, w_start:w_start+kw, :] += (
                    W[:, :, :, k]*dZ[:, i, j, k][:, None, None, None])
                dW[:, :, :, k] += np.sum(
                    region*dZ[:, i, j, k][:, None, None, None], axis=0)

    # drop the padded border for same padding
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
