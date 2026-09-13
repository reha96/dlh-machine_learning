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
    pass
