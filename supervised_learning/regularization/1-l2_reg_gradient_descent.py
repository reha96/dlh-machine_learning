#!/usr/bin/env python3
"""Gradient descent with L2 regularization (numpy)."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Y is a one-hot numpy.ndarray of shape (classes, m) containing the
    correct labels for the data.
    weights is a dictionary of the weights and biases of the neural
    network.
    cache is a dictionary of the outputs of each layer of the neural
    network.
    alpha is the learning rate.
    lambtha is the L2 regularization parameter.
    L is the number of layers of the network.

    Updates weights in place. Returns: None.
    """
    # gradient descent with L2 is first shrink then step
    # alpha is step size, lambtha is shrinkage strength (same as in cost)
    # we update W with penalty, without Biases

    # first we need m = number of examples
    # Y is (classes, m), same m as cache A0 is (784, m)
    m = Y.shape[1]

    # start backprop at last layer
    # for softmax+cross-entropy, dZ_last = A_L - Y
    # (vs regression residual y - yhat)

    # loop backwards L..1 to propagate error
    for layer in range(L, 0, -1):
        # get activations
        # A_prev is (n_{l-1}, m), A_curr is (n_l, m)
        # W is (n_l, n_{l-1}) — row = dest neuron
        A_prev = cache['A' + str(layer - 1)]
        A_curr = cache['A' + str(layer)]

        # compute dZ for this layer
        # last layer already has dZ, hidden uses tanh derivative
        # tanh' = 1 - A^2 (vs sigmoid A*(1-A) in earlier tasks)

        # compute gradients
        # dW = (1/m) * dZ dot A_prev.T + lambtha/m * W  — L2 extra term
        # db = (1/m) * sum(dZ, axis=1, keepdims)        — no L2
        # Why /m not /2m? /2 cancelled in derivative d/dW 1/2||W||2 = W

        # save copy of W before update — need W.T for dA_prev

        # update weights in place
        # W -= alpha * dW
        # b -= alpha * db

        # propagate to previous layer if not at input
        # dA_prev = W.T dot dZ
        # dZ_prev = dA_prev * (1 - A_prev**2)

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot


if __name__ == '__main__':
    lib = np.load(
        '/home/rehat/Documents/GitHub/dlh-machine_learning/supervised_learning/error_analysis/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = {}
    cache['A0'] = X_train
    cache['A1'] = np.tanh(
        np.matmul(weights['W1'], cache['A0']) + weights['b1'])
    cache['A2'] = np.tanh(
        np.matmul(weights['W2'], cache['A1']) + weights['b2'])
    Z3 = np.matmul(weights['W3'], cache['A2']) + weights['b3']
    cache['A3'] = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)
    print(weights['W1'])
    l2_reg_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.1, 3)
    print(weights['W1'])
