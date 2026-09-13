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
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    s_h, s_w = stride

    # calculate the output dimensions
    # start with padding
    if padding == "valid":
        ph, pw = 0, 0
    else:
        ph = ((h_prev-1)*s_h + kh - h_prev)//2
        pw = ((w_prev-1)*s_w + kw - w_prev)//2

    
    
    
    
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load(
        '/home/rehat/Documents/GitHub/dlh-machine_learning/supervised_learning/classification/data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    def relu(Z):
        return np.maximum(Z, 0)

    plt.imshow(X_train[0])
    plt.show()
    A = conv_forward(X_train_c, W, b, relu, padding='valid')
    print(A.shape)
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
