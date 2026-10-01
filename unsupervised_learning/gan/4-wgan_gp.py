#!/usr/bin/env python3
# (Based on 3-generate_faces.py)
"""WGAN-GP face generator with pretrained weights (GANs task 4)."""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """Wasserstein GAN with gradient penalty, loaded from disk.

    Same class as in task 2, extended with replace_weights to recover
    a model trained for 150 epochs by the project developers.
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
        # call super().__init__() first, store every argument,
        # set beta_1=.3 and beta_2=.9, store lambda_gp, then derive
        # the helpers listed on the intranet (dims, len_dims, axis,
        # scal_shape), fill in the generator and discriminator
        # losses, create the Adam optimizers and compile both
        # networks
        pass

    def replace_weights(self, gen_h5, disc_h5):
        """Replace the weights of both networks by stored ones.

        Args:
            gen_h5: File (.h5) where the generator weights are
                stored.
            disc_h5: File (.h5) where the discriminator weights are
                stored.
        """
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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Interpolate between real and fake samples.

        Args:
            real_sample: Batch of real samples.
            fake_sample: Batch of fake samples.

        Returns:
            The batch of samples interpolating real and fake.
        """
        pass

    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty on interpolated samples.

        Args:
            interpolated_sample: Batch of interpolated samples.

        Returns:
            The mean squared deviation of the gradient norm from 1.
        """
        pass

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
        # for _ in range(self.disc_iter):
        #     in a tape watching the discriminator's weights, get a
        #     real, a fake and an interpolated sample, compute the
        #     loss discr_loss and the gradient penalty gp, then the
        #     sum new_discr_loss = discr_loss + self.lambda_gp * gp
        #     apply gradient descent once on new_discr_loss
        # in a tape watching the generator's weights, get a fake
        # sample and compute the loss gen_loss of the generator
        # apply gradient descent to the generator
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss,
        #         "gp": gp}
        pass
