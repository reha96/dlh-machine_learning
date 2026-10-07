# GANs

Generative Adversarial Networks in TensorFlow/Keras: adversarial and
Wasserstein training loops, a gradient-penalty critic, convolutional
generator/discriminator networks for 16x16 faces, and a pretrained
face generator recovered from `.h5` weights.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | What a GAN is and why generator and discriminator are trained adversarially |
| 2 | What a latent vector and a latent space are |
| 3 | What Wasserstein GANs change and why the critic is not a classifier |
| 4 | How weight clipping and the gradient penalty enforce the Lipschitz constraint |
| 5 | How to build a convolutional generator/discriminator for images |
| 6 | How to save and reload network weights with Keras `.h5` files |

---

## Task-by-Task Reference

### Task 0 — Simple GAN (`0-simple_gan.py`)

**Challenge:** Implement the adversarial game itself: a Keras model whose
`train_step` updates two separate networks with two different losses,
drawing fresh real and fake batches on every step.

**Approach:** The generator maps latent vectors to samples and is pushed
toward making the discriminator output 1 on fakes (MSE against ones). The
discriminator is pushed toward 1 on real samples and -1 on fakes. Each
`train_step` performs `disc_iter` discriminator updates and then one
generator update, each inside its own `tf.GradientTape`; real batches are
drawn by shuffling indices, fake batches by sampling latent vectors.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `keras.Model` subclass with custom `train_step` | Take over the training loop entirely |
| `tf.GradientTape` + `optimizer.apply_gradients` | Manual gradient descent for two players |
| `tf.random.shuffle` / `tf.gather` | Draw a random batch of real examples |
| `tf.keras.losses.MeanSquaredError` | Score outputs against the target 1 / -1 |
| `disc_iter` | Let the discriminator keep up with the generator |

> **Key takeaway:** A GAN is just two networks trained alternately on
> opposing objectives — the "adversarial" part is the sign of the losses.

---

### Task 1 — Wasserstein GAN with clipping (`1-wgan_clip.py`)

**Challenge:** Replace the saturating MSE game by the Wasserstein critic
objective and keep the critic 1-Lipschitz by clipping its weights.

**Approach:** The generator loss becomes the negative mean of the critic
on fakes; the discriminator loss becomes mean(critic(fake)) minus
mean(critic(real)) — the players now collaborate to estimate a distance
rather than to classify. After every discriminator update all of its
trainable weights are clipped into [-1, 1] with `tf.clip_by_value`, which
is the cheap way to bound the critic's slope.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `-tf.reduce_mean(x)` generator loss | Move fakes toward higher critic scores |
| `mean(fake) - mean(real)` critic loss | Estimate the Wasserstein distance |
| `weight.assign(tf.clip_by_value(...))` | Enforce the Lipschitz constraint |
| `trainable_weights` loop | Clip weights and biases together |

> **Key takeaway:** Removing the sigmoid and clipping the critic's weights
> gives gradients that do not vanish when real and fake distributions are
> far apart.

---

### Task 2 — WGAN with gradient penalty (`2-wgan_gp.py`)

**Challenge:** Drop clipping (which caps the critic's capacity) and
penalize the deviation of the critic's gradient norm from 1 instead.

**Approach:** The class derives the data shape once (`dims`, `len_dims`,
`axis`, `scal_shape`) so it can build one random interpolation weight per
sample. `get_interpolated_sample` returns a convex combination of a real
and a fake batch; `gradient_penalty` differentiates the critic with
respect to that interpolated batch and returns the mean squared deviation
of the gradient norm from 1. The discriminator minimizes
`discr_loss + lambda_gp * gp`, while the returned dict still reports the
raw `discr_loss` alongside `gen_loss` and `gp`. Optimizers use
`beta_1=.3`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `tf.random.uniform(self.scal_shape)` | One interpolation coefficient per sample |
| Nested `tf.GradientTape` + `tape.watch` | Differentiate the critic w.r.t. its input |
| `tf.reduce_sum(..., axis=self.axis)` | Norm over every non-batch dimension |
| `lambda_gp=10` | Weight of the penalty term |

> **Key takeaway:** Penalizing the gradient norm is a softer, more
> expressive Lipschitz constraint than clipping, and it needs no bound on
> the weights.

---

### Task 3 — Convolutional generator and discriminator (`3-generate_faces.py`)

**Challenge:** Move from dense toy clouds to 16x16 greyscale faces, which
requires convolutional up/down-sampling and a bounded output.

**Approach:** `convolutional_GenDiscr` returns two functional Keras models.
The generator projects the 16-dim latent vector to 2048 units, reshapes it
to 2x2x512, then applies three upsampling blocks (UpSampling2D, Conv2D
with `padding="same"`, BatchNormalization, tanh) shrinking 64 -> 16 -> 1
channels until it reaches 16x16x1. The discriminator mirrors it with four
conv/max-pool/tanh blocks (32 -> 64 -> 128 -> 256 filters), then Flatten
and a single tanh output. Every activation, including the Dense layers, is
tanh so both networks output values in (-1, 1).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `keras.Input` + functional API | Build multi-layer image models |
| `UpSampling2D` | Grow 2x2 feature maps to 16x16 |
| `BatchNormalization` | Stabilize the generator's deep stack |
| `Conv2D(..., padding="same")` | Keep spatial dimensions predictable |
| `tanh` everywhere | Match the normalized [-1, 1] image data |

> **Key takeaway:** The generator upsamples coarse feature maps into an
> image; the discriminator downsamples the image back to a single score.

---

### Task 4 — Pretrained face generator (`4-wgan_gp.py`)

**Challenge:** Recover a WGAN-GP trained for 150 epochs by the project
developers, whose weights are stored on disk, and play with it.

**Approach:** The task-2 class is extended with `replace_weights(gen_h5,
disc_h5)`, which loads the stored `.h5` weight files into the existing
generator and discriminator with `load_weights` (topological order, no
`by_name`). Once the weights are in place the trained generator can
produce fake faces, and the critic can rank them.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.load_weights(path)` | Restore trained weights into a freshly built model |
| `.h5` weight files | Portable storage of all layer weights |
| Pretrained-model workflow | Separate architecture code from training cost |

> **Key takeaway:** Architecture and weights are independent: build the
> same network, then load the stored weights to recover a trained model
> without retraining.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | Custom `train_step` with two optimizers and `tf.GradientTape` | Adversarial training |
| 1 | Wasserstein losses and weight clipping with `tf.clip_by_value` | Wasserstein GAN |
| 2 | Interpolated samples and critic gradient penalty | Wasserstein GAN |
| 3 | Conv/UpSampling/BatchNorm generator, conv/max-pool discriminator | Convolutional models |
| 4 | `load_weights` from `.h5` to recover a pretrained model | Model persistence |

---

## Resources

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs (Gulrajani et al., 2017)](https://arxiv.org/abs/1704.00028)
- [Introduction to the Wasserstein distance (video)](https://www.youtube.com/watch?v=CDiol4LG2Ao)
- Detailed per-resource summaries and quiz hooks: see `RESOURCES.md`
