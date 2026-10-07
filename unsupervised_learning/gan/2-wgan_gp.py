#!/usr/bin/env python3
# (Based on 1-wgan_clip.py)
"""Wasserstein GAN with gradient penalty (GANs task 2)."""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """Wasserstein GAN replacing clipping by a gradient penalty.

    The two losses are the same as in the WGAN_clip class. The
    discriminator loss is penalized by lambda_gp times the gradient
    penalty of interpolated samples. Adam optimizers use beta_1=.3 and
    beta_2=.9.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """Initialize the WGAN-GP model.

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
            lambda_gp (int, optional): Coefficient of the gradient
                penalty (default 10).
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
        self.beta_1 = .3
        # standard value, but can be changed if necessary
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        # derive the shapes needed to interpolate samples
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Interpolate between real and fake samples.

        Args:
            real_sample: Batch of real samples.
            fake_sample: Batch of fake samples.

        Returns:
            The batch of samples interpolating real and fake.
        """
        # one random weight per sample and its complement
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty on interpolated samples.

        Args:
            interpolated_sample: Batch of interpolated samples.

        Returns:
            The mean squared deviation of the gradient norm from 1.
        """
        # watch the interpolated sample and differentiate the critic
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        # the penalty is the squared deviation of the gradient norm from 1
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """Run one training step of the WGAN-GP.

        Apply disc_iter gradient descents on the penalized
        discriminator loss, then one gradient descent to the
        generator.

        Args:
            useless_argument: Data argument passed by Keras; unused,
                samples are drawn from the stored attributes.

        Returns:
            A dict with the "discr_loss", "gen_loss" and "gp"
            values.
        """
        # train the discriminator disc_iter times
        for _ in range(self.disc_iter):
            real = self.get_real_sample()
            fake = self.get_fake_sample()
            # interpolate between the real and fake samples
            interpolated = self.get_interpolated_sample(real, fake)
            # compute the losses in a tape watching the discriminator
            with tf.GradientTape() as tape:
                pred_real = self.discriminator(real, training=True)
                pred_fake = self.discriminator(fake, training=True)
                discr_loss = self.discriminator.loss(pred_real, pred_fake)
                gp = self.gradient_penalty(interpolated)
                # penalize the discriminator loss by the gradient penalty
                new_discr_loss = discr_loss + self.lambda_gp * gp
            # apply one gradient descent to the discriminator
            grads = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))
        # train the generator once
        with tf.GradientTape() as tape:
            fake = self.get_fake_sample(training=True)
            pred_fake = self.discriminator(fake, training=False)
            gen_loss = self.generator.loss(pred_fake)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
