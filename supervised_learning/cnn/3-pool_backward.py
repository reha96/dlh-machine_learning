#!/usr/bin/env python3
"""Backward propagation over a pooling layer."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network.

    Args:
        dA (numpy.ndarray): of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the output of the
            pooling layer.
            - m is the number of examples
            - h_new is the height of the output
            - w_new is the width of the output
            - c_new is the number of channels
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_new)
            containing the output of the previous layer.
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
        kernel_shape (tuple): (kh, kw) containing the size of the kernel
            for the pooling.
            - kh is the kernel height
            - kw is the kernel width
        stride (tuple): (sh, sw) containing the strides for the pooling.
            - sh is the stride for the height
            - sw is the stride for the width
        mode (str): either 'max' or 'avg', indicating whether to perform
            maximum or average pooling, respectively.

    Returns:
        numpy.ndarray: the partial derivatives with respect to the previous
        layer (dA_prev).
    """
    # extract relevant shapes
    # partial derivatives (dA) obs, height, width, channels
    m, h_new, w_new, c_new = dA.shape
    # input (A) obs, height, width, channels
    m, h_prev, w_prev, c = A_prev.shape
    # kernel (kh, kw) height, width
    kh, kw = kernel_shape
    # step size height, width
    sh, sw = stride

    # init the output dimensions
    dA_prev = np.zeros(A_prev.shape)

    # sweep each output position and scatter the gradients back
    for i in range(h_new):
        for j in range(w_new):
            # zone of the input covered by the filter
            h_start = i*sh
            w_start = j*sw
            zone = A_prev[:, h_start:h_start+kh, w_start:w_start+kw, :]

            # scatter gradients for max or avg pooling
            if mode == "max":
                mask = (zone == np.max(zone, axis=(1, 2), keepdims=True))
                dA_prev[:, h_start:h_start+kh, w_start:w_start+kw, :] += (
                    mask*dA[:, i, j, :][:, None, None, :])
            else:
                dA_prev[:, h_start:h_start+kh, w_start:w_start+kw, :] += (
                    dA[:, i, j, :][:, None, None, :]/(kh*kw))

    return dA_prev
