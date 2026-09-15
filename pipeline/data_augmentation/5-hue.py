#!/usr/bin/env python3
"""Change the hue of an image."""
import tensorflow as tf


def change_hue(image, delta):
    """Change the hue of an image.

    Args:
        image: 3D tf.Tensor containing the image to change.
        delta: amount the hue should change.

    Returns:
        The altered image.
    """
    return tf.image.adjust_hue(
    image, delta, name=None
)
