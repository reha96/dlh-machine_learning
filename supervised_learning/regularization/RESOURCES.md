# Regularization — Resources (intranet project 2297)

Ingested 2026-09-05. Status key: `summary` = full treatment done;
`paywalled` = paywall survived the chain (title + first paragraph only);
`failed-transcript` = captions unavailable, watch manually; `deferred` =
explicitly postponed. References are link-only entries. Both Medium
articles returned HTTP 403 to bots with no usable archive copy and were
read through the authenticated browser session instead (full text
accessible, summarized below in own words). Transcripts were pulled with
`youtube-transcript-api` and kept in `/tmp` (never committed).

## Read or watch — Articles

### Regularization (mathematics)

URL: https://en.wikipedia.org/wiki/Regularization_(mathematics)  ·  Date: 2026-08-15 (last edited)  ·  Status: summary

The general mathematical frame behind every task in this project:
an ill-posed or overfitting-prone fit is stabilized by adding a
penalty (explicit regularization) or by altering the algorithm itself
(implicit regularization such as early stopping or dropout). The page
connects penalties to Bayesian priors, works through Tikhonov/ridge
regression in closed form, and surveys sparsity (L1/LASSO, elastic
net), group penalties, and early stopping as regularization in time.

- Explicit = extra cost term (priors, penalties, constraints); implicit = everything else (early stopping, robust losses, SGD itself).
- Classifier objective form: min over f of empirical loss plus λ·R(f); λ trades data fit against simplicity (Occam's razor).
- Tikhonov/ridge: squared L2 penalty, differentiable, closed-form solution with an extra λnI in the inverse.
- L1 induces sparsity (convex stand-in for the NP-hard L0 count); elastic net mixes L1+L2 and groups correlated features.
- Early stopping = regularization in time: iteration count caps complexity, with validation-based rules for practice.

### An Overview of Regularization Techniques in Deep Learning

URL: https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/  ·  Date: 2018-04 (publication; page last updated 2025-05-01)  ·  Status: summary

Hands-on survey of the four techniques behind tasks 0–7 (read up to,
but excluding, the MNIST-with-Keras case study per the spec): L1/L2
penalties added to the cost, dropout as random node removal, data
augmentation as cheap extra examples, and early stopping on a
validation split. Each comes with the Keras one-liner used in
practice, which maps directly onto the TF tasks in this project.

- L2 (ridge/weight decay) shrinks weights toward but never to zero; L1 (lasso) can zero weights out, so it compresses models.
- Dropout reads as training a different thinned network per iteration — an ensemble effect; drop probability is the hyperparameter.
- Augmentation (flip, rotate, scale, shift via ImageDataGenerator) manufactures invariance the dataset lacks.
- Early stopping via callbacks watches a validation metric with a patience count; overshooting patience wastes epochs, undershooting stops a recovery.
- Practical note echoed across answers: tune λ (or patience) on held-out data, never on the test set.

### L2 Regularization and Back-Propagation

URL: https://jamesmccaffreyblog.com/2017/02/19/l2-regularization-and-back-propagation/  ·  Date: 2017-02-19 (publication)  ·  Status: summary

Short practitioner piece on exactly what task 1 implements: the L2
penalty enters the error term, its derivative adds a λ·w term to the
gradient, and the update becomes shrink-then-step, which is why the
method is also called weight decay. Most useful for its warnings about
the details that silently break implementations.

- Overfit networks show large-magnitude weights; the penalty rewards small ones.
- Update order: first scale the weight by (1 − η·λ/n), then subtract η times the backprop gradient.
- Biases are NOT regularized — their update equation is unchanged, an easy bug when hand-coding.
- Coding shortcut (plain step, then subtract a fraction of the weight) changes what λ means; keep the constant's definition consistent.
- With non-gradient optimizers (e.g. particle swarm) the penalty must enter the error directly instead.

### Intuitions on L1 and L2 Regularisation

URL: https://medium.com/data-science/intuitions-on-l1-and-l2-regularisation-235f2db4c261  ·  Date: 2018-12-26 (publication)  ·  Status: summary

Builds L1/L2 intuition purely from the gradient-descent update on a
one-variable regression: the L1 penalty shifts each step by a constant
±λ (sign-driven, pushes weights exactly to zero → sparsity and feature
removal), while the L2 penalty shifts by a term proportional to the
current weight (magnitude-driven, shrinks everything smoothly). Four
plain-language framings of why any such penalty helps: it moves the
solution away from the overfit optimum, trades perfect training fit
for generality, injects data-independent information (λ), and —
for L1 — deletes useless variables.

- L1 = lasso, squared-L2 = ridge; both are constraints the optimizer must respect alongside fitting y.
- L1 update depends on the sign of w; L2 update depends on sign, magnitude, and twice the λ — neither dominates universally.
- Sparsity mechanism: positive weights get λ subtracted, negative ones get λ added, so both drift to exactly zero.
- Too-large λ underfits severely; λ is tuned, never derived.
- The one-variable derivation extends unchanged to deep nets (matrices instead of scalars).

### Analysis of Dropout

URL: https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/  ·  Date: 2017-01-10 (publication)  ·  Status: summary

The theoretical companion to tasks 4–6: dropout trains an ensemble of
thinned networks whose predictions are averaged, and its core benefit
is preventing neurons from co-adapting into brittle correction cliques.
Single-neuron masks are Bernoulli trials, so a layer's dropped count is
Binomial (the reason the spec points at `numpy.random.binomial`);
inverted dropout (scale by 1/keep_prob during training, untouched test
pass) is the framework-standard variant because it needs no test-time
rescaling. Also derives why dropout pairs with L2: inverted scaling
boosts the effective learning rate by 1/q, so an unconstrained net can
diverge without a weight penalty alongside.

- Per-neuron mask ~ Bernoulli(p); per-layer drop count ~ Binomial(n, p) with mean np.
- Direct (test-time scaling by q) vs inverted (train-time scaling by 1/q) dropout; frameworks use inverted.
- Without the 1/q correction, downstream layers see shrunken activations at train time and saturate/explode at test time.
- Inverted dropout inflates the effective step to η/q — combine with L2/max-norm to keep selection of η sane.
- Dropout alone cannot stop weights growing; that is L2's job, so the two are routinely stacked.

### Early stopping

URL: https://en.wikipedia.org/wiki/Early_stopping  ·  Date: 2026-04-23 (last edited)  ·  Status: summary

Treats stopping time as the regularization knob: gradient methods drift
from simple to complex fits, so halting when validation error bottoms
out caps complexity as surely as a penalty term. Covers the
statistical-learning view (iteration budgets with generalization
bounds, boosting consistency) and the practical holdout recipe that
task 7 encodes — split off validation data, check it periodically,
stop after patience checks with no real improvement, and keep the best
weights rather than the last ones.

- Split train/validation (e.g. 2:1); validation error proxies generalization error.
- Naive rule: stop at the first validation uptick and roll back one checkpoint.
- Real curves are noisy with local minima, so patience + improvement threshold replace the naive rule.
- Task-7 vocabulary maps here: threshold = "real" improvement, patience = checks tolerated, count = checks so far.
- Same source the wiki cites for rules (Prechelt, "Early Stopping — But When?") is a project reference below.

### How to use early stopping properly for training deep neural network?

URL: https://stats.stackexchange.com/questions/231061/how-to-use-early-stopping-properly-for-training-deep-neural-network  ·  Date: 2016-08-22 (asked)  ·  Status: summary

Q&A on the practical doubts behind task 7: how often to validate,
whether to skip early epochs, and how to survive a bouncing validation
curve. Consensus: validate every epoch (cheap when validation is
small), do not skip epochs (nobody knows the right skip count),
absorb the bounce with patience (commonly 10–100, often 10–20), and
always restore the best weights, not the weights where training
halted. A second answer adds the hindsight alternative — over-train
once, checkpoint often, pick the best afterwards — when compute allows.

- Validation frequency: each epoch is the common default.
- Early epochs can look worse before converging; patience covers that without special-casing.
- Patience = epochs to wait with no progress; typical 10–20, dataset-dependent.
- Keep best (patience-epochs-ago) weights; stopping weights are by definition past-peak.
- First pointer given is Prechelt's paper — the same reference the spec lists.

### Data Augmentation | How to use Deep Learning when you have Limited Data

URL: https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced  ·  Date: 2018-04-11 (publication)  ·  Status: summary

Motivates augmentation as regularization-by-data for the blog task
(task 8): networks with millions of parameters starve on small
datasets, and label-preserving transforms (flip, rotate, scale, crop,
translate, noise) multiply effective data while killing spurious
cues — illustrated by a classifier that "learns" car brand from facing
direction. Distinguishes offline augmentation (pre-expand small
datasets) from online/on-the-fly augmentation (transform mini-batches,
for large data), catalogs interpolation modes for out-of-bounds
pixels, and demos the payoff (76% → 94.5% on a 50-images-per-class
task; DenseNet/CIFAR table). Warns that only plausible transforms
help: upside-down cars regularize nothing for a road camera.

- Big models need proportional data; augmentation manufactures it nearly free.
- Invariance (translation, viewpoint, scale, illumination) is taught, not innate.
- Offline (small data, fixed expansion factor) vs online (large data, per-batch, GPU-friendly).
- Boundary fill matters: constant, edge, reflect, symmetric, wrap.
- Irrelevant transforms add irrelevant data — match augmentation to deployment conditions.

## Read or watch — YouTube (deeplearning.ai Course 2)

### Regularization (C2W1L04)

URL: https://www.youtube.com/watch?v=6g0t3Phly2M  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Derives the L2-regularized cost for logistic regression then scales it
to networks: add λ/2m times the squared norm (Frobenius norm for
matrices), leave the single bias b out (one parameter among thousands
changes nothing), tune λ on a dev set, and note the `lambtha`
spelling (lambda is a Python keyword). The gradient gains a λ/m·W
term, so each update multiplies W by (1 − αλ/m) — slightly below one —
which is the entire content of the "weight decay" nickname.

- Logistic: J + (λ/2m)·‖w‖²; network: J + (λ/2m)·Σ‖W[l]‖²_F over layers.
- L1 variant (λ/m·‖w‖₁) yields sparse W but is rarely used for nets; L2 dominates practice.
- Frobenius norm = sum of squared entries; the /2m scaling keeps the gradient clean.
- Debug tip: the MONITORED cost must include the penalty, or monotonic decrease checks mislead.

### Why Regularization Reduces Overfitting (C2W1L05)

URL: https://www.youtube.com/watch?v=NyG-7nRpsW8  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Two intuitions for why shrinking weights generalizes: a heavily
penalized net drives many weights near zero, collapsing toward a
smaller, logistic-like network that cannot express wild boundaries;
and with tanh activations, small weights keep pre-activations in the
near-linear regime, so even a deep stack behaves almost linearly and
stays simple. Recommends trying an intermediate λ that lands between
the underfit and overfit extremes.

- Large λ → near-zero W → effectively fewer active hidden units → simpler model.
- Small W → small Z → tanh ≈ linear → deep net ≈ linear function, low capacity.
- Zeroing-out is approximate: all units stay, each matters less.
- Practical: implement it once and watch variance drop on the exercise curves.

### Dropout Regularization (C2W1L06)

URL: https://www.youtube.com/watch?v=D8PJAL-MZv8  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Implements inverted dropout line by line: per-layer keep probability
(0.8 in the example), boolean mask from uniform draws compared
against keep_prob, element-wise zeroing, then division by keep_prob to
preserve expected activations. Masks are redrawn every iteration and
also gate the backward pass; at test time dropout is off entirely, and
the 1/keep_prob scaling is what makes that sound. Each example trains
a different thinned network, hence the ensemble/regularization effect.

- keep_prob per layer (1.0 = layer skipped); 0.5–0.8 typical for hidden layers.
- Mask recipe: uniform random array, threshold at keep_prob, multiply, divide by keep_prob.
- New masks each iteration and each pass; forward and backward share the mask.
- Test time: no masks, no extra scaling — predictions stay deterministic.
- Cost J is ill-defined under dropout (masks change every step); disable it (keep_prob=1) to verify monotonic decrease first.

### Understanding Dropout (C2W1L07)

URL: https://www.youtube.com/watch?v=ARq74QuavAo  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Explains why random deletion regularizes: no unit can lean on any
single input (it may vanish), so weights spread across features, which
shrinks their squared norm — an adaptive, per-weight form of L2.
Advises per-layer keep probabilities aimed where the parameters live
(big matrices get low keep_prob, input layers stay near 1.0), notes
dropout's computer-vision roots (huge inputs, perpetual overfitting),
and repeats the debugging caveat: never gradient-check with dropout on.

- Cannot-rely effect → spread weights → smaller ‖W‖², like L2 but input-adaptive.
- keep_prob is per-layer; more parameters in a layer → stronger dropout there.
- Input-layer dropout is rare and gentle (≈0.9–1.0); deleting half the inputs is seldom wanted.
- Cost of flexibility: one more hyperparameter per regularized layer.
- Use dropout only against actual overfitting; it is not a default-on ingredient elsewhere.

### Other Regularization Methods (C2W1L08)

URL: https://www.youtube.com/watch?v=BOCLq2gpcGU  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Two remaining tools for the blog task: data augmentation (flips,
rotations, zooms, distortions as nearly-free extra training examples
that teach invariance — horizontal flips yes, upside-down cats no)
and early stopping (halt when dev error bottoms out). Frames early
stopping's strength (one training run sweeps small→large weight norms,
no λ grid search) against its cost: it breaks orthogonalization by
mixing "optimize the cost" with "don't overfit" into a single knob,
so the speaker defaults to L2 with a λ search when compute allows.

- Augmentation = cheap data; distortions must preserve the label and plausibly occur.
- Early stopping exploits small-init weights growing over time; stopping early picks mid-size norms.
- Advantage: single run explores the whole capacity path; no λ sweep needed.
- Downside: couples optimization and regularization (orthogonalization violated).
- Preference stated: L2 + λ search if affordable, early stopping otherwise.

### deeplearning.ai (course site)

URL: https://www.deeplearning.ai/  ·  Date: unknown  ·  Status: verified (link-only; homepage, no summarizable content — the five videos above are its substance)

## References

Link-only entries; HTTP status verified via `curl -sI -L` on 2026-09-05
unless noted. Last-Modified dates recorded where the header is
meaningful (numpy/TF doc pages); PDF file dates are server artifacts,
marked unknown.

- R1. [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) — verified HTTP 200 · Date: 2026-06-28 (page last modified) · Status: verified
- R2. [numpy.random.binomial](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html) — verified HTTP 200 · Date: unknown (no Last-Modified header) · Status: verified
- R3. [tf.keras.regularizers.L2](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L2) — verified HTTP 200 · Date: 2024-06-07 (page last modified) · Status: verified
- R4. [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) — verified HTTP 200 · Date: 2024-06-07 (page last modified) · Status: verified
- R5. [Regularization loss (why does model.losses return regularization losses)](https://stackoverflow.com/questions/56693863/why-does-model-losses-return-regularization-losses) — bot-challenged (HTTP 403 to curl and headless browser; loads in interactive browsers) · Date: unknown · Status: verified-in-browser (URL resolves, Cloudflare gate)
- R6. [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) — verified HTTP 200 · Date: 2024-06-07 (page last modified) · Status: verified
- R7. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR paper, PDF)](http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) — verified HTTP 200 · Date: unknown · Status: verified
- R8. [Early Stopping — But When? (Prechelt, PDF)](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf) — verified HTTP 200 · Date: unknown · Status: verified
- R9. [L2 Regularization versus Batch and Weight Normalization (arXiv PDF)](https://arxiv.org/pdf/1706.05350) — verified HTTP 200 · Date: unknown · Status: verified

## Quiz Hooks

- Regularization — stabilizing a fit (penalty term or algorithm change) to cut generalization error.
- Overfitting signature — huge weight magnitudes; the penalty rewards small ones.
- L2 cost (logistic) — J + (λ/2m)·‖w‖²; network version sums Frobenius norms over layers.
- Frobenius norm — sum of squared entries of a matrix; the matrix analogue of squared L2.
- lambtha — λ renamed because lambda is a Python keyword.
- Weight decay — L2's alias: each update multiplies W by (1 − αλ/m), slightly below one.
- Biases excluded — single b among thousands of weights; regularizing it changes nothing.
- L1 vs L2 — L1 shifts steps by ±λ (sign-driven, exact zeros, sparsity); L2 scales by weight (smooth shrinkage).
- Lasso / ridge — linear models with L1 / squared-L2 penalties.
- Elastic net — L1+L2 mix; groups correlated features.
- Dropout — per-iteration random thinned network; ensemble effect without the ensemble cost.
- keep_prob — per-layer probability of keeping a unit; 1.0 skips the layer.
- Inverted dropout — divide by keep_prob at train time; test time untouched and deterministic.
- Mask distribution — per-neuron Bernoulli, per-layer count Binomial with mean n·p.
- Dropout + L2 — inverted scaling boosts the effective rate to η/q, so pair them.
- Co-adaptation — neurons covering for each other; what dropout breaks.
- Data augmentation — label-preserving transforms as cheap data; offline (small sets) vs online (large sets).
- Early stopping — stopping time as capacity knob; halt at dev-error minimum.
- Patience / threshold / count — tolerated flat checks / minimum real improvement / flat checks so far.
- Restore best weights — stopping weights are past-peak by definition.
- Orthogonalization — one tool per job; early stopping mixes optimization with regularization.
- model.losses — per-layer penalty terms collected from regularized layers; added to the data cost.
- Explicit vs implicit — added penalty term vs algorithm-side effects (stopping, dropout, SGD noise).
