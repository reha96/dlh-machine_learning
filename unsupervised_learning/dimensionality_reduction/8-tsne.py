#!/usr/bin/env python3
"""Write a function that performs a t-SNE transformation
    """
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
X is a numpy.ndarray of shape (n, d) containing the dataset
ndims is the new dimensional representation of X
idims is the intermediate dimensional representation after PCA
perplexity is the perplexity
iterations is the number of iterations
lr is the learning rate

Returns:
Y, a numpy.ndarray of shape (n, ndim) containing the optimized low
dimensional transformation of X
    """
    # step 1: drop X to idims dimensions with PCA first
    X = pca(X, idims)
    n, d = X.shape
    # step 2: build the P affinities and exaggerate them for 100 iterations
    P = P_affinities(X, perplexity=perplexity)
    P = P * 4
    # step 3: start Y at random and keep the previous step for momentum
    Y = np.random.randn(n, ndims)
    Y_prev = np.zeros((n, ndims))
    # step 4: optimize Y with gradient descent and momentum
    for i in range(1, iterations + 1):
        dY, Q = grads(Y, P)
        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8
        Y_new = Y - lr * dY + momentum * (Y - Y_prev)
        Y_prev = Y
        Y = Y_new
        # step 5: keep Y centered on the origin
        Y = Y - np.mean(Y, axis=0)
        # step 6: stop exaggerating after iteration 100
        if i == 100:
            P = P / 4
        # step 7: report the cost every 100 iterations
        if i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

    return Y
