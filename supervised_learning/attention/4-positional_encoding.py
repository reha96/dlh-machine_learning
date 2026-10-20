#!/usr/bin/env python3
"""Positional encoding (attention task 4)."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculate the positional encoding for a transformer.

    Args:
        max_seq_len (int): maximum sequence length.
        dm (int): model depth.

    Returns:
        np.ndarray: shape (max_seq_len, dm) containing the positional
            encoding vectors.
    """
    PE = np.zeros((max_seq_len, dm))

    # One row per position, one frequency per dimension pair
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(dm // 2)[np.newaxis, :]
    angles = positions / (10000 ** ((2 * dims) / dm))

    # Sines on the even columns, cosines on the odd ones
    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles)

    return PE
