#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network: the model to train
        data: a numpy.ndarray of shape (m, nx) containing the input data
        labels: a one-hot numpy.ndarray of shape (m, classes) containing the
            labels of data
        batch_size (int): the size of the batch used for mini-batch gradient
            descent
        epochs (int): the number of passes through data for mini-batch
            gradient descent
        verbose (bool): determines if output should be printed during
            training
        shuffle (bool): determines whether to shuffle the batches every epoch

    Returns:
        the History object generated after training the model
    """
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle)
