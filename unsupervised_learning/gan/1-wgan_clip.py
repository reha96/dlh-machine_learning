#!/usr/bin/env python3
# (Based on 0-simple_gan.py)
"""Wasserstein GAN with weight clipping (GANs task 1)."""

import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """Wasserstein GAN where the two players collaborate.

    generator_loss(x) is the opposite of the mean value of the image
    x of the discriminator on the image by the generator of a batch of
    latent vectors. discriminator_loss(x, y) compares the mean value
    of the discriminator on the real batch y with the mean value on
    the fake batch x. The weights of the discriminator must be
    clipped in [-1, 1].
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the WGAN model.

        Args:
            generator: The generator network.
            discriminator: The discriminator network.
            latent_generator: Function generating a batch of latent
                vectors of the requested size.
            real_examples: Real examples used for training.
            batch_size (int, optional): Number of samples per batch.
            disc_iter (int, optional): Number of discriminator updates
                per training step.
            learning_rate (float, optional): Learning rate of both
                Adam optimizers.
        """
        # call super().__init__() first, store every argument,
        # set beta_1=.5 and beta_2=.9,
        # new: fill in the generator loss with tf.reduce_mean,
        # fill in the discriminator loss with tf.reduce_mean, then
        # create the Adam optimizers and compile both networks
        pass

    def get_fake_sample(self, size=None, training=False):
        """Generate a batch of fake samples.

        Args:
            size (int, optional): Number of samples to generate.
                Defaults to the batch size.
            training (bool, optional): Flag passed to the generator
                to select training or inference mode.

        Returns:
            The batch of fake samples produced by the generator.
        """
        pass

    def get_real_sample(self, size=None):
        """Draw a random batch from the real examples.

        Args:
            size (int, optional): Number of samples to draw.
                Defaults to the batch size.

        Returns:
            A batch of real samples drawn uniformly at random.
        """
        pass

    def train_step(self, useless_argument):
        """Run one training step of the Wasserstein GAN.

        Apply disc_iter gradient descents to the discriminator,
        clipping its weights in [-1, 1] after each descent, then one
        gradient descent to the generator.

        Args:
            useless_argument: Data argument passed by Keras; unused,
                samples are drawn from the stored attributes.

        Returns:
            A dict with the "discr_loss" and "gen_loss" values.
        """
        # for _ in range(self.disc_iter):
        #     get a real sample and a fake sample
        #     in a tape watching the discriminator's weights, compute
        #     the loss discr_loss on the real and fake samples
        #     apply gradient descent once to the discriminator
        #     clip the weights of the discriminator between -1 and 1
        # in a tape watching the generator's weights, get a fake
        # sample and compute the loss gen_loss of the generator
        # apply gradient descent to the generator
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
        pass
