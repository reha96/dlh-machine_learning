#!/usr/bin/env python3
"""Makes a prediction using a neural network."""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose (bool): determines if output should be printed during the
            prediction process

    Returns:
        the prediction for the data
    """
    # predict is forward propagation, with the same math as evaluate
    # but without labels and without scoring:
    # return model's output tensor
    return network.predict(x=data, verbose=verbose)
