#!/usr/bin/env python3
"""Bag of words embedding (word_embeddings task 0)."""

import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix.

    Args:
        sentences (list): list of sentences to analyze.
        vocab (list): list of the vocabulary words to use for the
            analysis. If None, all words within sentences should be
            used.

    Returns:
        tuple: (embeddings, features)
            embeddings (np.ndarray): shape (s, f) containing the
                embeddings; s is the number of sentences in sentences,
                f is the number of features analyzed.
            features (list): the list of features used for embeddings.
    """
    pass
