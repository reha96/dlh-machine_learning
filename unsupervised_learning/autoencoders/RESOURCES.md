# Autoencoders Resources

Ingested: 2026-09-25

## Autoencoder — definition (Neural networks [6.1])

URL: https://www.youtube.com/watch?v=FzS3tMl4Nsc&t=73s · Date: 2013-11-16 (YouTube upload) · Status: summary

An autoencoder is a feed-forward network trained without labels to reproduce its input. Its encoder maps the input into a latent code and its decoder uses that code to reconstruct the input; a narrower code can force the model to preserve the most useful information.

- The target is the original input, so input and reconstruction have matching dimensions and meaning.
- The latent representation is the encoder's output and can be used as a compact description of the input.
- The decoder's output activation should match the input range or data type.
- Tying decoder weights to the transpose of encoder weights is an option, not a requirement.

## Autoencoder — loss function (Neural networks [6.2])

URL: https://www.youtube.com/watch?v=xTU79Zs4XKY · Date: 2013-11-16 (YouTube upload) · Status: summary

The reconstruction objective compares the original sample with its output and is chosen to suit the data. The lesson derives Bernoulli cross-entropy for binary inputs and squared error for real-valued inputs, then shows how ordinary backpropagation trains the encoder and decoder.

- Binary-valued inputs commonly use cross-entropy with a sigmoid reconstruction output.
- Real-valued inputs can use half the squared Euclidean reconstruction error and a linear output.
- For these paired choices, the output pre-activation gradient reduces to reconstruction minus input.
- With tied weights, gradients from both the encoder and decoder contribute to the shared parameters.
- More structured observations can be handled by selecting an output probability model and minimizing its negative log-likelihood.

## Deep learning — deep autoencoder (Neural networks [7.6])

URL: https://www.youtube.com/watch?v=z5ZYm_wJ37c · Date: 2013-11-16 (YouTube upload) · Status: summary

A deep autoencoder gradually compresses data through several encoder layers and expands it through a decoder to reconstruct the input. Training both deep halves together can be difficult; the lecture describes layer-wise unsupervised pretraining as one way to improve initialization, while deeper nonlinear decoding can represent reconstructions beyond a linear PCA model.

- A narrow middle representation can compress data or provide a two-dimensional view for visualization.
- Greedy pretraining was shown to help optimization in the examples, even when the issue was underfitting rather than overfitting.
- A multilayer nonlinear decoder is not constrained by PCA's linear reconstruction result.
- Better optimization methods are another route to training deep encoder-decoder networks.

## Introduction to autoencoders — Jeremy Jordan

URL: https://www.jeremyjordan.me/autoencoders/ · Date: 2018-03-19 (publication date) · Status: summary

This article presents autoencoders as representation-learning models whose reconstruction objective and constrained latent representation encourage discovery of structure in unlabeled data. It contrasts several ways to avoid a model that merely memorizes its training examples and notes uses such as denoising, anomaly detection, inpainting, and information retrieval.

- An undercomplete model uses a narrow hidden code; nonlinear networks can learn structure beyond a linear PCA subspace.
- Sparse variants penalize hidden activations, for example with an L1 term or a KL penalty on average activation rates.
- Denoising variants receive corrupted inputs but are trained against clean targets, encouraging recovery of underlying structure.
- Contractive variants penalize encoder sensitivity to small input changes through a Jacobian-based term.
- Reconstruction quality must be balanced against regularization so the code remains useful without simply memorizing examples.

## Variational Autoencoders — EXPLAINED! (to 12:55)

URL: https://www.youtube.com/watch?v=fcvYpzHmhvA · Date: 2019-06-17 (YouTube upload) · Status: summary

This video motivates variational autoencoders as generative models. A plain autoencoder can reconstruct training examples but may leave gaps or irregular regions in its latent codes, so random sampling can decode to implausible outputs. A VAE constrains the learned latent distributions to a continuous region that can be sampled and explored.

- The encoder-decoder pair first learns to reconstruct input samples through a compact code.
- A VAE represents codes probabilistically, making it possible to sample from a known prior region instead of guessing arbitrary latent vectors.
- Reconstruction loss preserves input detail, while a latent-distribution penalty encourages codes to occupy a smoother, connected space.
- Sampling nearby points or interpolating in that space can produce plausible variations rather than outputs from untrained gaps.

## Variational Autoencoders — Arxiv Insights

URL: https://www.youtube.com/watch?v=9zKuYvjFFS8 · Date: 2018-02-25 (YouTube upload) · Status: summary

The video develops the VAE objective and training mechanics from the conventional encoder-decoder model. Instead of emitting one fixed bottleneck vector, the VAE predicts distribution parameters, samples a code, and combines reconstruction quality with a penalty that keeps the learned latent distribution near a standard normal prior.

- The encoder predicts a mean and standard deviation; the decoder reconstructs from a sample drawn using them.
- The KL term regularizes the latent distribution toward a zero-mean, unit-variance Gaussian, alongside reconstruction loss.
- Reparameterization writes the sample as `z = μ + σ ⊙ ε`, with `ε` sampled from a fixed standard normal, allowing gradients to train `μ` and `σ`.
- A beta-VAE changes the relative strength of the KL penalty to encourage more separated latent factors, with a possible loss of reconstruction detail when the penalty is too strong.

## Intuitively Understanding Variational Autoencoders — Irhum Shafkat

URL: https://medium.com/data-science/intuitively-understanding-variational-autoencoders-1bfe67eb5daf · Date: 2018-02-04 (publication date) · Status: summary

The article explains that a reconstruction-trained autoencoder does not necessarily organize its latent space so arbitrary samples or interpolations decode well. A VAE instead learns distributions for its codes and regularizes those distributions toward a shared prior, encouraging a latent space that supports generation and smooth exploration.

- A plain autoencoder's latent codes may form disconnected regions, leaving the decoder unprepared for samples in between.
- The VAE samples codes from distributions parameterized by encoder-produced means and standard deviations.
- KL divergence toward a standard normal prior encourages a dense, continuous latent space, while reconstruction loss retains input information.
- Sampling, interpolation, and vector offsets in latent space can be used to explore generated variations.

## Deep Generative Models — Actian

URL: https://www.actian.com/glossary/deep-generative-models/ · Date: unknown · Status: summary

This overview describes deep generative models as neural systems that learn patterns in data and use those patterns to synthesize new examples. It places VAEs among several model families and emphasizes their continuous latent representations as a way to sample and create smooth variations.

- VAEs learn latent representations that support reconstruction and generation of related samples.
- Other families listed include GANs, autoregressive models, normalizing flows, energy-based models, and score-based models.
- Applications span synthetic images, text and speech, along with augmentation and other data-generation uses.
- The page has no visible byline publication date, so its date is recorded as unknown.

## Definitions to skim

- [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) — an asymmetric measure of how one probability distribution differs from another; it is zero when the distributions match.
- [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) — a neural network that encodes an input into a representation and decodes it to reconstruct the input.
- [Generative model](https://en.wikipedia.org/wiki/Generative_model) — a model that captures patterns or a probability distribution in data so it can describe or generate plausible examples.

## References

- [The Deep Learning textbook — Chapter 14: Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html)
- [Reducing the Dimensionality of Data with Neural Networks (2006)](https://www.cs.toronto.edu/~hinton/absps/science.pdf)

## Quiz Hooks

- Encoder and decoder — the two mappings that create a latent representation and reconstruct an input.
- Bottleneck — a capacity constraint that encourages a compact code.
- Reconstruction loss — a measure of the difference between an input and its reconstruction.
- Binary cross-entropy vs. squared error — common reconstruction objectives for binary vs. real-valued inputs.
- Kullback–Leibler divergence — a distribution-comparison penalty used to regularize latent codes.
- Reparameterization — express a random latent sample using learned parameters and separate fixed noise so gradients can flow.
- Continuous latent space — a property that makes VAE sampling and interpolation useful for generation.
