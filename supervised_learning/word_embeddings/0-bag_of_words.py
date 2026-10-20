#!/usr/bin/env python3
"""Bag of words embedding (word_embeddings task 0)."""

import re
import string

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
    # Normalize every sentence, then split it into words
    tokenized = []
    for sentence in sentences:
        text = sentence.lower()
        # Possessive 's goes first, otherwise "children's" -> "childrens"
        text = re.sub(r"\'s\b", "", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        tokenized.append(text.split())

    # Vocabulary: sorted when built here, caller order when supplied
    if vocab is None:
        vocab = sorted({word for words in tokenized for word in words})

    features = np.array(vocab)
    columns = {word: i for i, word in enumerate(vocab)}

    # Count every surviving word into its column
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, words in enumerate(tokenized):
        for word in words:
            if word in columns:
                embeddings[i, columns[word]] += 1

    return embeddings, features
