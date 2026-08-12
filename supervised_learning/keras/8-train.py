#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent and analyzes validation
data."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True,
                shuffle=False):
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
        learning_rate_decay (bool): indicates whether learning rate decay
            should be used
        alpha (float): the initial learning rate
        decay_rate (float): the decay rate
        save_best (bool): indicates whether to save the model after each
            epoch if it is the best
        filepath (str): the file path where the model should be saved
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
    # append and also save the best iteration of the model
    if save_best and filepath is not None:
        callbacks.append(
            K.callbacks.ModelCheckpoint(filepath=filepath,
                                        monitor='val_loss',
                                        save_best_only=True))

    # no need to calculate learning rate lr from scratch, we need to pass
    # a function that does this and Keras will handle the rest
    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)
        callbacks.append(
            K.callbacks.LearningRateScheduler(scheduler, verbose=1))

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
