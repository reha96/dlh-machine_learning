#!/usr/bin/env python3
"""Perform a random crop of an image."""
import tensorflow as tf


def crop_image(image, size):
    """Perform a random crop of an image.

    Args:
        image: 3D tf.Tensor containing the image to crop.
        size: tuple containing the size of the crop.

    Returns:
        The cropped image.
    """
    # take a random crop of the requested size
    return tf.image.random_crop(image, size)
