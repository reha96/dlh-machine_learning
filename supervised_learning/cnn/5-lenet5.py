#!/usr/bin/env python3
"""LeNet-5 architecture built with Keras."""
from tensorflow import keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using keras.

    The model consists of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes

    All layers requiring initialization initialize their kernels with the
    he_normal initialization method, with the seed set to zero for each
    layer to ensure reproducibility. All hidden layers requiring activation
    use the relu activation function.

    Args:
        X (K.Input): of shape (m, 28, 28, 1) containing the input images
            for the neural network.
            - m is the number of images

    Returns:
        K.Model: the compiled model to use Adam optimization (with default
        hyperparameters) and accuracy metrics.
    """
    pass
