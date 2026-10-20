#!/usr/bin/env python3
"""Train a Word2Vec model (word_embeddings task 2)."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build and train a gensim word2vec model.

    Args:
        sentences (list): list of sentences to be trained on.
        vector_size (int): dimensionality of the embedding layer.
        min_count (int): minimum number of occurrences of a word for
            use in training.
        window (int): maximum distance between the current and
            predicted word within a sentence.
        negative (int): size of negative sampling.
        cbow (bool): True is for CBOW, False is for Skip-gram.
        epochs (int): number of iterations to train over.
        seed (int): seed for the random number generator.
        workers (int): number of worker threads to train the model.

    Returns:
        the trained model.
    """
    # gensim names the objective by its inverse: sg=0 is CBOW, sg=1 skip-gram
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   negative=negative,
                                   seed=seed,
                                   workers=workers,
                                   sg=sg)

    # Training runs once in the constructor and once here
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
