# GANs Resources

Ingested: 2026-09-29

## Generative Adversarial Networks — Goodfellow et al. (2014)

URL: https://arxiv.org/pdf/1406.2661 · Date: 2014-06-10 (arXiv submission) · Status: summary

The founding GAN paper. It proposes estimating generative models through an
adversarial game: a generator G learns the data distribution while a
discriminator D estimates whether a sample comes from the data or from G, and
both are trained at the same time. G tries to maximize the chance that D makes
a mistake; the framework is a minimax two-player game. In the space of
arbitrary functions a unique equilibrium exists where G reproduces the data
distribution and D outputs 1/2 everywhere. When G and D are multilayer
perceptrons the whole system trains with plain backpropagation — no Markov
chains or unrolled approximate inference. Experiments cover digit
classification and generative modeling of MNIST, Toronto Face Dataset and
CIFAR-10 samples.

- Two models trained jointly: generator G (latent → data) and discriminator D
  (data → probability of being real).
- Training alternates between maximizing D's accuracy and fooling D; D's
  sigmoid output ranges over [0,1].
- The value function is `E[log D(x)] + E[log(1 − D(G(z)))]`; the paper notes
  minimizing `−log D(G(z))` works better in practice for G (vanishing
  gradients of the saturating variant).
- No explicit density needed: only backprop through G and D, sampling from
  the latent distribution (uniform or normal) suffices.
- Caveat raised by the paper itself: training can be unstable and the
  equilibrium is only guaranteed in theory for the idealized infinite-capacity
  case.

## Wasserstein GAN — Arjovsky et al. (2017)

URL: https://arxiv.org/pdf/1701.07875 · Date: 2017-01-26 (arXiv submission) · Status: summary

Replaces the usual GAN objective with the Earth-Mover (Wasserstein-1)
distance between the generated and real distributions. The point is that this
distance stays meaningful when the two supports do not overlap, unlike
KL/JS-style divergences that saturate and starve the generator of gradient.
The discriminator becomes a "critic" trained to estimate the distance, which
must be kept 1-Lipschitz — here enforced by clipping its weights to a small
box after every update. The result is stabler training, disappearance of mode
collapse in the reported experiments, and critic losses that correlate with
sample quality, so the loss curve becomes a usable debugging signal.

- Wasserstein distance = minimal "work" to reshape one distribution into the
  other; it metrizes convergence of distributions.
- Kantorovich-Rubinstein duality: the W1 distance equals the supremum of
  `E[f(x)] − E[f(y)]` over 1-Lipschitz functions f — this is exactly what the
  critic optimizes.
- The critic has no sigmoid and is trained several steps per generator step;
  the generator loss is simply minus the critic's output on fake samples
  (the "opposite of the mean" recipe used in this project's tasks).
- Lipschitz constraint via weight clipping: simple but crude (later shown by
  Gulrajani et al. to cause capacity problems).
- WGAN objective lower bounds the KL divergence, connecting it to older
  divergences and explaining why the new loss is better behaved.

## Improved Training of Wasserstein GANs — Gulrajani et al. (2017)

URL: https://arxiv.org/pdf/1704.00028 · Date: 2017-03-31 (arXiv submission) · Status: summary

Diagnoses why WGAN's weight clipping hurts: clipping either under- or
over-shapes the critic's weights, giving it too little or too much capacity,
and the clipping threshold is a delicate hyperparameter. The paper replaces
clipping with a gradient penalty: sample points uniformly along the straight
line between a real and a generated sample, and add `λ · (mean(‖∇ critic‖ − 1)²)`
to the critic loss so the critic has unit gradient norm along those lines.
The method trains stably across very different architectures with almost no
hyperparameter tuning — 101-layer ResNets, convolutional GANs on CIFAR-10 and
LSUN bedrooms, and language models over discrete data — and produces
higher-quality samples than clipped WGANs.

- This project's task 2 (WGAN_GP) is a direct implementation: interpolation
  between real and fake, gradient penalty, `lambda_gp=10`.
- Interpolated points are drawn on the segment real → fake (one point per
  example), exactly the `get_interpolated_sample`/`gradient_penalty` methods
  of the task.
- The penalty replaces clipping entirely — nothing is clipped in WGAN_GP.
- Practical caveats: the critic should not use batch normalization (penalizes
  the wrong thing); layer normalization is suggested instead; penalty weight
  λ still needs coarse tuning.
- Empirical claim: generator quality tracks critic loss far better with GP
  than with clipping.

## Introduction to the Wasserstein distance — Applied Algebraic Topology Network

URL: https://www.youtube.com/watch?v=CDiol4LG2Ao · Date: 2020-10-26 (YouTube upload) · Status: video-transcript-summary

A ~18-minute lecture introducing the Wasserstein-1 distance from scratch,
with the aliases it carries: Kantorovich-Rubinstein, optimal transport, and
earth mover's distance. It starts from an intuition contrast with the
sup-norm: two functions that are horizontal translates look maximally
different pointwise, yet require almost no "work" to turn one into the other.
The construction then moves to measures with finite support visualized as
piles of mass, where a transport plan says how much mass moves from each
source pile to each target pile, and the distance is the cheapest such plan.

- The transport plan is a joint distribution on pairs of points whose
  marginals must reproduce the two input measures — no mass invented or
  destroyed.
- Cost of a plan: for each matched pair, distance moved times amount of mass
  moved; the Wasserstein distance is the infimum of that cost over all plans.
- Finite-support formula the video computes by hand; general formula is an
  integral of pairwise distance against the coupling measure π.
- Geodesics intuition: under the sup-norm the path between two functions
  distorts shape; under Wasserstein it slides mass, preserving shape.
- Why GANs care: W1 keeps giving informative gradients even when generated
  and real supports are disjoint — the reason the WGAN paper picks it.

## This person does not exist

URL: https://thispersondoesnotexist.com/ · Date: unknown · Status: stub (domain for sale)

The rltoken now resolves to a bare "domain for sale" page (verified
2026-09-30; the 2026-09-28 archive.org capture shows the same sale page), so
there is no live site to summarize and no summary is offered. The project
page describes the resource's original purpose: a page that served one
photograph of a synthetic person per refresh — the demo of what GAN face
generators can produce, which task 4 rebuilds from a trained WGAN_GP.
Historical captures still hold the original behavior: archive.org snapshots
from 2019-02 (image/png) and 2023-05-31 (image/webp) serve a single face
image directly.

- Destination changed after the intranet link was written; do not expect a
  working demo when explaining task 4.
- Wayback Machine preserves the face-serving era (e.g.
  https://web.archive.org/web/20230531212903/https://thispersondoesnotexist.com/).
- Practical substitute for the demo: generate faces locally with the
  pretrained weights of task 4, or any public StyleGAN face generator.

## Quiz Hooks

- GAN — generator and discriminator trained adversarially until the
  generator's samples pass as real.
- Latent vector / latent space — small random input the generator maps
  continuously to a data sample; nearby codes give similar samples.
- Minimax game — the training objective: one player's gain is the other's
  loss; equilibrium at D ≡ 1/2.
- Critic — sigmoid-free discriminator in WGANs whose scalar output scores
  how "real" a sample looks.
- Wasserstein-1 (earth mover's) distance — cheapest transport cost between
  two distributions; stays informative when supports do not overlap.
- Lipschitz constraint — bounds the critic's slope; enforced by weight
  clipping (WGAN) or a gradient penalty (WGAN-GP).
- Gradient penalty — mean squared deviation of the critic's input-gradient
  norm from 1 along real↔fake interpolates, weighted by lambda_gp.
- Weight clipping — clamp critic weights into [-1, 1] after each update;
  simple but capacity-limiting.
- Mode collapse — generator produces a narrow range of samples; a classic
  failure the Wasserstein objective mitigates.
- MSE GAN loss — this project's Simple_GAN scores fakes against 1 and reals
  against 1 / fakes against −1 with mean squared error.
