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
        super().__init__()                         # run the __init__ of Keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        # standard value, but can be changed if necessary
        self.beta1 = .5
        # standard value, but can be changed if necessary
        self.beta2 = .9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=self.beta1, beta_2=self.beta2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=self.beta1, beta_2=self.beta2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

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
        if size is None:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Draw a random batch from the real examples.

        Args:
            size (int, optional): Number of samples to draw.
                Defaults to the batch size.

        Returns:
            A batch of real samples drawn uniformly at random.
        """
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:self.batch_size]
        return tf.gather(self.real_examples, random_indices)

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
        for _ in range(self.disc_iter):
            #     get a real sample and a fake sample
            #     in a tape watching the discriminator's weights, compute
            # jj     the loss discr_loss on the real and fake samples
            real = self.get_real_sample()
            fake = self.get_fake_sample()
            #     apply gradient descent once to the discriminator
            # in a tape watching the generator's weights, get a fake
            # sample and compute the loss gen_loss of the generator
            with tf.GradientTape() as tape:
                pred_real = self.discriminator(real, training=True)
                pred_fake = self.discriminator(fake, training=True)
                discr_loss = self.discriminator.loss(pred_real, pred_fake)
            # apply gradient descent to the generator
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))
            # generator
            with tf.GradientTape() as tape:               # a NEW tape (G's)
                # inside => G's vars watched
                fake = self.get_fake_sample(training=True)
                p_fake = self.discriminator(
                    fake, training=False)   # opponent frozen
                # one-arg lambda, INSIDE tape
                gen_loss = self.generator.loss(p_fake)
            grads = tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(
                zip(grads, self.generator.trainable_variables))
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
