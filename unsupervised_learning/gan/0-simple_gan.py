#!/usr/bin/env python3
# (Based on N/A — Task 0 has no preceding task skeleton.)
"""Simple Generative Adversarial Network (GANs task 0)."""

import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """A GAN played by two adversary players.

    The generator loss is the mean squared error between the output of
    the discriminator on a fake batch and 1. The discriminator loss is
    the mean squared error between its output on a fake batch and -1,
    summed with the mean squared error between its output on a real
    batch and 1. Both networks are compiled with Adam optimizers
    (beta_1=.5, beta_2=.9).
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the GAN model.

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
        # and optimizer and the discriminator loss and optimizer

        # call super().__init__()
        # inheriting from keras.Model, it will 
        # initialize the Keras model's internal state
        super().__init__()

        # store every argument
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        # set betas
        self.beta_1 = .5
        self.beta_2 = .9

        # define the generator loss
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        
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
        """Run one training step of the GAN.

        Apply disc_iter gradient descents to the discriminator, then
        one gradient descent to the generator.

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
        # in a tape watching the generator's weights, get a fake
        # sample and compute the loss gen_loss of the generator
        # apply gradient descent to the generator
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
        pass
