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
        # run the __init__ of Keras.Model first and store the arguments
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        # standard value, but can be changed if necessary
        self.beta_1 = .5
        # standard value, but can be changed if necessary
        self.beta_2 = .9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.reduce_mean(y) - tf.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
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
        if not size:
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
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

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
        # train the discriminator disc_iter times
        for _ in range(self.disc_iter):
            real = self.get_real_sample()
            fake = self.get_fake_sample()
            # compute the loss in a tape watching the discriminator
            with tf.GradientTape() as tape:
                pred_real = self.discriminator(real, training=True)
                pred_fake = self.discriminator(fake, training=True)
                discr_loss = self.discriminator.loss(pred_real, pred_fake)
            # apply one gradient descent to the discriminator
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))
            # clip the weights of the discriminator between -1 and 1
            for weight in self.discriminator.trainable_weights:
                weight.assign(tf.clip_by_value(
                    weight, clip_value_min=-1, clip_value_max=1))
        # train the generator once
        with tf.GradientTape() as tape:
            fake = self.get_fake_sample(training=True)
            pred_fake = self.discriminator(fake, training=False)
            gen_loss = self.generator.loss(pred_fake)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
