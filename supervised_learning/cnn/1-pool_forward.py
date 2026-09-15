#!/usr/bin/env python3
"""Forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
            - m is the number of examples
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
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
        numpy.ndarray: the output of the pooling layer.
    """
    # extract relevant shapes
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    s_h, s_w = stride

    # calculate output dimensions
    output_h = (h_prev - kh)//s_h + 1
    output_w = (w_prev - kw)//s_w + 1
    A_pooled = np.zeros((m, output_h, output_w, c_prev))

    # sweep each filter across the input
    for i in range(output_h):
        for j in range(output_w):
            # region of the input covered by the filter
            h_start = i*s_h
            w_start = j*s_w
            region = A_prev[:, h_start:h_start+kh, w_start:w_start+kw, :]

            # aggregate (pooling)
            if mode == "max":
                A_pooled[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                A_pooled[:, i, j, :] = np.mean(region, axis=(1, 2))

    return A_pooled
