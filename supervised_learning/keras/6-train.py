#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent and analyzes validation
data."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and analyzes validation
    data.

    Args:
        network: the model to train
        data: a numpy.ndarray of shape (m, nx) containing the input data
        labels: a one-hot numpy.ndarray of shape (m, classes) containing the
            labels of data
        batch_size (int): the size of the batch used for mini-batch gradient
            descent
        epochs (int): the number of passes through data for mini-batch
            gradient descent
        validation_data: the data to validate the model with, if not None
        early_stopping (bool): indicates whether early stopping should be
            used
        patience (int): the patience used for early stopping
        verbose (bool): determines if output should be printed during
            training
        shuffle (bool): determines whether to shuffle the batches every epoch

    Returns:
        the History object generated after training the model
    """

    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks = [K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)]

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
