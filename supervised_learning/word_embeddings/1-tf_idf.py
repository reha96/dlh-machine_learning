#!/usr/bin/env python3
"""TF-IDF embedding (word_embeddings task 1)."""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding.

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
    # A None vocabulary is learned from the corpus and comes back sorted;
    # a supplied vocabulary keeps the order the caller gave us.
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
