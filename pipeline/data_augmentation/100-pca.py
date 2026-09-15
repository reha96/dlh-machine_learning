#!/usr/bin/env python3
"""Perform PCA color augmentation."""
import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation as in the AlexNet paper.

    Args:
        image: 3D tf.Tensor containing the image to change.
        alphas: tuple of length 3 containing the amount that each
            channel should change.

    Returns:
        The augmented image.
    """
    # save the input shape (h, w, c) for the final reshape
    orig_shape = tf.shape(image)
    # scale to 0 to 1 range (float for the math)
    flat_norm = tf.cast(image, tf.float32) / 255.0

    # flatten pixels out as rows of RGB values (N, 3)
    flat_pixels = tf.reshape(flat_norm, (-1, 3))
    # get the mean color of the image (1, 3)
    color_mean = tf.reduce_mean(flat_pixels, axis=0, keepdims=True)
    # remove the mean to center the colors
    centered = flat_pixels - color_mean

    # count how many pixels (N) the image has
    num_pixels = tf.cast(tf.shape(flat_pixels)[0], tf.float32)

    # build the 3 by 3 color covariance matrix (channels vs channels)
    covar = tf.matmul(tf.transpose(centered), centered) / num_pixels

    # split the covariance into directions (vecs) and strengths (vals)
    eig_vals, eig_vecs = tf.linalg.eigh(covar)
    # scale each strength by its random amount (alphas)
    scaled_vals = eig_vals * alphas

    # make the scaled strengths a column vector (3, 1)
    scaled_col = tf.reshape(scaled_vals, [-1, 1])
    # mix the directions by the scaled strengths (3, 1)
    delta_vec = tf.matmul(eig_vecs, scaled_col)
    # turn the color shift into a row vector (1, 3)
    delta_row = tf.transpose(delta_vec)
    # copy the one shift to every pixel (N, 3)
    delta_full = tf.broadcast_to(delta_row, tf.shape(centered))

    # add the shift back onto the centered colors
    shifted = centered + delta_full + color_mean
    # keep every value inside the 0 to 1 range
    clipped = tf.clip_by_value(shifted, 0, 1)
    # fold the pixels back into an image (h, w, c)
    restored = tf.reshape(clipped, orig_shape)
    # go back to 0 to 255 integer pixels
    return tf.cast(restored * 255, tf.uint8)
