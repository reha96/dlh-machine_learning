# Optimization — 20-Question MCQ (self-test)

> **Source-grounded:** definitions and formulas match `RESOURCES.md`
> and the course notebook at
> `https://notebooklm.google.com/notebook/9e3bcac0-163a-465a-8167-63a63ad00dc7`
> (standardization, batch norm, mini-batch, EWMA/momentum, RMSProp,
> Adam, LR decay).
> **How to use:** Try all 20 without looking at
> `quiz_optimization_answers.md`. Pick one letter per question.
> **Level:** undergrad-accessible. Math appears where it helps
> (`RESOURCES.md:105-112` for optimizer formulas), but each stem can be
> answered from intuition alone.
> **Design:** Within each topic block questions go high → low:
> Q1 = what/why (purpose, intuition), Q2 = core mechanics,
> Q3 = formulas, hyperparameters and edge cases. All four options per
> question are length-balanced (≈ ±15% chars) and equally detailed so
> length cannot hint at the answer. Correct positions are randomized
> (≈ 5× each of A/B/C/D).

---

## Feature Scaling — 3Q (high → low)

### Q1. [Feature scaling — why it matters]
You train a neural net to predict `house_price` from `income`
(20k–200k dollars) and `schooling_years` (8–20). Without scaling,
what happens to gradient descent?

A) Gradient descent is scale-invariant like a tree that splits on
thresholds, so `income` and `schooling` contribute equally without
scaling; scaling only matters for tree models, so you can skip it
for gradient-based learners entirely.

B) Scaling cures heteroskedasticity in OLS residuals after training;
unequal variance biases the gradient estimate, so the fix is
statistical not geometric, and optimization speed is unchanged by
feature ranges or units.

C) The wide dollar range dominates distances and stretches the cost
surface into a narrow ravine, so steps zig-zag across steep walls
and need tiny alpha; scaling rounds the contours so updates point
more directly toward the minimum.

D) The narrow-range feature must get the larger weight, so the net
always overfits `schooling_years` and ignores `income`; scaling
worsens that imbalance by amplifying the small scale, hurting
generalization by design.

### Q2. [Feature scaling — standardization vs min-max]
Which pairing of method, formula and effect is correct?

A) Standardization: `x' = (x − min)/(max − min)` → [0,1], robust to
outliers by design; it clips heavy tails and is the neural-net
default because it bounds every feature to the same interval.

B) Standardization (z-score): `x' = (x − mean)/std` → mean 0,
variance 1; the usual default for SVM, logistic regression and
neural nets, making contours rounder without imposing hard bounds.

C) Min-max normalization: `x' = (x − mean)/std` → exactly [0,1];
it bounds data and is always better than standardization when
outliers exist because the mean and std resist extreme values.

D) Both methods are identical in practice and the standard min-max
formula is `gamma·x + beta`; the two names just describe the
learnable affine step later used inside batch normalization layers.

### Q3. [Feature scaling — practice, heavy tails and train/test]
In econometrics you often log a heavy-tailed regressor (population,
GDP per capita) before scaling and must respect train/test splits.
Which workflow is correct?

A) Compute a fresh mean and std on the test set so each split is
perfectly zero-mean unit-variance on its own; this maximizes fit
on test and mirrors how population GDP is rebased each forecast.

B) Always apply min-max to [0,1] on test even if train used
standardization; the test interval must be fixed to [0,1] by
construction, regardless of the train transform you chose earlier.

C) Skip scaling entirely if you plan to use batch normalization
later; batch norm inside hidden layers already normalizes inputs,
so input scaling is redundant and only adds extra computation.

D) Compute `mean` and `std` on train only, then apply
`(X − mean_train)/std_train` to both train and test; if a feature
is very skewed, log-transform it first so skew does not dominate
the scaling.

---

## Batch Normalization — 3Q (high → low)

### Q4. [Batch norm — purpose and intuition]
What does batch normalization do at a high level and why does it
let very deep nets train faster?

A) It pins each layer's pre-activation mean/variance per
mini-batch, decoupling layers so deeper layers see stable inputs
(less covariate shift, black-cat→colored-cat); allows larger
learning rates, helps very deep nets train, and adds mild
dropout-like noise from batch statistics.

B) It adjusts the global learning rate at fixed intervals to avoid
over-adapting; by shrinking alpha on a schedule it acts as
learning-rate decay, prevents late overshoot, and mimics temporal
shrinkage of step size across epochs.

C) It deletes the learnable `gamma` and `beta` to shrink the
model; fewer parameters mean faster forward passes and less memory,
which is claimed as the main reason deeper nets converge more
quickly despite no change in activation distributions.

D) It always slows training because mean/variance computations per
batch dominate the cost; wall-clock time strictly increases and the
tiny regularization effect never offsets overhead, so net training
time is claimed to always get worse.

### Q5. [Batch norm — core mechanics]
For a hidden layer you compute `z = W·a_prev + b`, then apply batch
norm **before** the activation during training. Which sequence is
correct?

A) `z_norm = (z − min)/(max − min)` → `z_tilde = z_norm − gamma/beta`
→ `a = g(z_tilde)`; keep bias `b` as usual and treat gamma/beta as
extra output-layer weights shared across mini-batches, normalizing
by range rather than variance.

B) Normalize gradients once per epoch by `gamma` and `beta` to avoid
over-adapting the descent direction; apply the same scalar rescaling
to every layer and leave activations `z` untouched during all
forward passes.

C) Compute `mu, sigma2` over the mini-batch → `z_norm = (z − mu)/
sqrt(sigma2 + eps)` → `z_tilde = gamma·z_norm + beta` →
`a = g(z_tilde)`; bias `b` is dropped because it cancels under mean
subtraction and `beta` replaces it; gamma/beta are (n^[l],1).

D) Batch norm is applied only to raw inputs `X`; hidden layers are
never normalized because their activations are already centered by
the previous non-linearity and need no extra stabilization step.

### Q6. [Batch norm — test time and details]
At inference you have a single example (no batch). How does batch
norm normalize?

A) Recompute `mu, sigma2` from that one example alone and force
`gamma = 0, beta = 0` for the forward pass; this centers the
single activation to zero exactly and avoids any running estimates.

B) Use the exponentially weighted running averages of `mu` and
`sigma2` collected during training, together with the learned
`gamma, beta`; frameworks store these estimates by default and
reuse them at test time.

C) Skip normalization at test time entirely and compute `a = g(z)`
directly; the stored statistics are only for monitoring and are
not used inside the normalize-rescale equations after training.

D) Re-estimate `mu, sigma2` on the entire test set before each
prediction and overwrite `gamma, beta` with those test statistics;
this recalibrates the layer to the test distribution every time.

---

## Mini-Batch Gradient Descent — 3Q (high → low)

### Q7. [Mini-batch — why it is the default]
Think of batch GD as a census (measure everyone before moving) and
SGD as one interview at a time. Why is mini-batch usually the
default?

A) It uses the whole dataset on every step, so the cost is never
noisy and the gradient is exact; this precision outweighs the
memory cost and makes it the most accurate optimizer per update.

B) It removes hyperparameter tuning entirely — batch size, learning
rate and momentum become irrelevant once you average over a batch,
so you can pick any values and still converge reliably and fast.

C) It guarantees escape from every saddle point without needing
momentum or Adam; the batch noise alone finds the global minimum
deterministically, even on high-dimensional non-convex surfaces.

D) It gives frequent progress before an epoch ends (like a
well-designed survey sample), keeps vectorized matrix ops within
the batch, reduces variance vs SGD, and stays cheaper and more
memory-friendly than full-batch GD.

### Q8. [Mini-batch — mechanics vs batch/SGD]
You have `m = 5,000,000` examples. How do update counts compare?

A) Batch GD: one update per **epoch** (cost over all `m`); SGD:
one per **example** (noisy, loses vectorization); mini-batch
(e.g., 256): one per **batch** (≈ 19,500 updates/epoch), still
vectorized inside each batch.

B) Mini-batch GD is the same as batch normalization: it normalizes
activations of a batch before each gradient step, so the two terms
describe the same operation on the same mini-batch of examples.

C) Batch GD does 5,000,000 updates per epoch; SGD does 256; mini-
batch with 256 does one update per epoch; the labels just swap
which regime is frequent versus infrequent per pass over data.

D) All three regimes do the same number of updates per epoch;
batch size only changes memory usage, not the count or frequency
of parameter updates or the geometry of optimization.

### Q9. [Mini-batch — cost curve and sizing details]
What cost behavior do you expect and how should you pick the batch
size?

A) Mini-batch cost must strictly decrease every iteration; the
batch size is fixed at 1000 by theory and should never be tuned
because the 1/(1−beta) window already controls all smoothing.

B) SGD has the smoothest cost curve and the best vectorization;
mini-batch is the noisiest and should be avoided because its
variance always prevents settling near any minimum.

C) Batch GD cost must decrease every iteration or your learning
rate is too large; mini-batch cost trends down but is noisy (some
batches harder); pick 32–512, power of 2, fits GPU memory; if
`m ≤ 2000`, plain batch GD is fine.

D) Batch size is itself learned by gradient descent jointly with
`W, b`; you optimize it as a parameter via back-prop so it adapts
automatically to the loss surface during training without manual
tuning.

---

## Momentum and EWMA — 3Q (high → low)

### Q10. [EWMA — what beta controls]
The EWMA `V_t = beta·V_{t-1} + (1−beta)·theta_t` with `V_0 = 0`
underlies momentum, RMSProp and Adam. What does `beta` control?

A) `beta` is the learning rate itself; larger `beta` means larger
gradient steps directly, so you raise `beta` to speed up and lower
it to stabilize, exactly like you would tune alpha.

B) `beta` sets memory: window ≈ `1/(1−beta)` steps; larger `beta`
(e.g., 0.98) is smoother but lags more, smaller (e.g., 0.5) is
responsive but noisy; weights decay as `(1−beta)·beta^k`.

C) `beta` counts how many past gradients are stored exactly; e.g.,
`beta = 10` means remember the last 10 batches verbatim and
`beta = 50` means keep 50 batches in a FIFO buffer.

D) `beta` must be 0 for any averaging to occur; `beta = 0.9`
disables smoothing entirely and returns the raw sequence
theta_t with no memory of previous values.

### Q11. [Momentum — update and ravine intuition]
Which update matches the Ng-style momentum used in this project,
and what does it do on an elongated bowl (ravine)?

A) `v_dW = dW / sqrt(s_dW + eps)` then `W ← W − alpha·v_dW`;
this divides by the second-moment estimate to shrink steep
directions, which is the definition of classic momentum.

B) `W ← W − alpha·dW + beta·W`; momentum adds weight decay
proportional to `W`; this regularizes large weights but does not
average past gradients or damp oscillations.

C) Momentum first normalizes every input feature to [0,1] before
each update; this rounding of contours removes the need for any
prior feature scaling or batch normalization step.

D) `v_dW = beta·v_dW + (1−beta)·dW;  W ← W − alpha·v_dW` (same for
`b`); on an elongated bowl it damps oscillation across steep walls
while accumulating progress along the flat floor, so larger alpha
is safe.

### Q12. [Momentum — hyperparameters and bias correction]
Practical momentum choices: defaults, variants and bias correction.

A) Default `beta=0.9` (≈10-step average) works; bias correction
`v/(1−beta^t)` exists but is usually skipped for momentum
(~10-step warm-up); variant `v=beta·v+dW` equals rescaling `alpha`,
yet changing `beta` then also rescales the step.

B) `beta` must be 0.999 and bias correction is mandatory or
momentum diverges; `epsilon=1e-8` is claimed as the momentum
coefficient and `beta` is just the numerical stability constant
inside that update.

C) Momentum is said to introduce `epsilon=1e-8` in a denominator to
avoid division by zero; `beta` is that `epsilon` and `epsilon` then
controls memory length exactly like `1/(1−beta)` would normally do.

D) Momentum should be used only with full-batch GD and never with
mini-batch GD; the noise in mini-batch gradients is said to break
averaging and make the method unstable on any batch size by itself
even with tuned alpha.

---

## RMSProp — 3Q (high → low)

### Q13. [RMSProp — purpose and intuition]
In what sense is RMSProp an adaptive learning-rate fix? What
per-dimension problem does it tame?

A) It keeps one global learning rate for all parameters and scales
updates by the signed mean gradient; every direction then gets the
same step, so volatile and quiet features are treated identically
despite very different curvatures.

B) It accumulates the simple sum `Sum dW^2` from the start without
decay; the denominator then grows forever and the effective rate
collapses to zero, which freezes learning late in training.

C) It gives each parameter its own effective rate: large,
oscillating gradients build a large running average of squares and
get smaller steps; quiet directions keep larger steps, curbing
vertical bounce while preserving motion on flat axes.

D) It learns `gamma` and `beta` to rescale activations per batch,
exactly like batch normalization, so the optimizer and the batch
normalizer are two names for the identical affine transform step.

### Q14. [RMSProp — core mechanics]
Which is the RMSProp update (per-parameter) and what does `s_dW`
track?

A) `s_dW = beta·s_dW + (1−beta)·dW` (no square), `W ← W −
alpha·s_dW`; `s_dW` tracks the mean gradient only, keeping sign
information and ignoring magnitude, then steps in that mean
direction.

B) `s_dW = beta·s_dW + (1−beta)·dW^2` (element-wise square), `W ←
W − alpha·dW / sqrt(s_dW + eps)`; `s_dW` is the EWMA of squared
gradients; steep or oscillating directions get larger `s` and thus
smaller effective steps.

C) RMSProp sums `dW^2` from the first iteration with no forgetting,
then divides by `sqrt(sum + eps)`; the denominator grows
monotonically and learning freezes late — the classic Adagrad
behavior it was meant to keep.

D) RMSProp learns `gamma` and `beta` to shift and scale the
normalized activation `z_norm`; it optimizes the batch-norm affine
parameters rather than the squared-gradient average per weight.

### Q15. [RMSProp — details, econ bridge and pros/cons]
Finance analogy: squared residuals show clustered volatility
(GARCH) with recent shocks weighted most. How is `s_dW` like that,
and what trade-off does RMSProp bring?

A) Both use a simple arithmetic mean over all history with equal
weights forever; RMSProp therefore never forgets old gradients and
treats a shock from epoch 1 the same as yesterday's gradient move.

B) Both require min-max scaling to [0,1] before any squaring; if
you do not bound inputs first, the square-and-average formula is
claimed to be invalid and the update is mathematically undefined.

C) GARCH and RMSProp both track the signed mean gradient, not the
square; they average raw `dW` with decay and step in the mean
direction, so magnitude information is intentionally discarded now.

D) Both use a decaying average of squared signals so recent large
magnitudes dominate; RMSProp shrinks steps where recent squared
gradients were large (volatile) and keeps steps larger where flat
— like weighting by realized volatility.

---

## Adam — 3Q (high → low)

### Q16. [Adam — purpose and behavior]
Adam keeps two EWMAs. At a high level, what does it combine and
why is its step bounded and scale-invariant?

A) It combines momentum (first moment) and RMSProp (second moment)
with bias correction; steps are bounded by `alpha` and largely
invariant to gradient rescaling, helping on plateaus and saddles;
correction matters early since `m,v` start at 0.

B) It uses only the second moment `v` of squared gradients and no
first moment or bias correction; the bound is said to come from
weight decay alone and the method then has no invariance to scale
at all.

C) It implements a linear learning-rate decay `alpha/(1+beta·t)`
and is just another name for inverse-time scheduling; the scale
invariance is claimed to come from shrinking alpha over epochs, not
from combining moments.

D) It learns batch-norm parameters `gamma·z_norm + beta` and calls
that Adam; the bound and invariance are then properties of
normalizing activations, not of combining two moment estimates at
all.

### Q17. [Adam — core mechanics and formulas]
Which equation set is Adam with bias correction (Kingma & Ba)?

A) `m = beta·m + g^2;  theta ← theta − alpha·m`; single second
moment with no correction; `m` accumulates squares forever and
`v` is never used or corrected.

B) `theta ← theta − alpha·g / (1 + beta·t)`; the linear decay of
alpha is Adam, with one global schedule shared across parameters
and no per-parameter moments at all.

C) `m = beta1·m + (1−beta1)·g;  v = beta2·v + (1−beta2)·g^2;
m_hat = m/(1−beta1^t);  v_hat = v/(1−beta2^t);  theta ← theta −
alpha·m_hat/(sqrt(v_hat) + eps)`.

D) `theta ← theta − alpha·g` after `gamma·z_norm + beta`; Adam
is the batch-norm affine rescaling followed by a plain gradient
step; moment estimates and eps play no role in the update.

### Q18. [Adam — defaults, cons and modern fixes]
Which facts about defaults, downsides and fixes are correct?

A) Adam has no usable defaults; you must heavily tune `beta1,
beta2, eps` every run while `alpha` hardly matters; the standard
is claimed to be `beta2=0.9`, `beta1=0.999` and correction is
`m·(1−beta^t)`.

B) Defaults `beta1=0.9, beta2=0.999, eps=1e-8` (some use `1e-7`);
tune `alpha` first; can generalize worse than SGD+momentum as it
forgets rare gradients; fixes: AMSGrad (max `v`), AdamW (decoupled
decay), SWATS (Adam→SGD).

C) Adam's only downside is extra memory for batch-norm `gamma,
beta`; it never generalizes worse and L2 regularization is said
to equal weight decay for every optimizer including Adam.

D) Adam diverges on any non-convex surface and should never be
tried; Ng is said to advise against it because plateaus and
saddles make the moment estimates unstable from the first step.

---

## Learning Rate Decay — 2Q (high → low)

### Q19. [LR decay — why and when vs adaptives]
You train with fixed `alpha = 0.1` and mini-batch loss wanders
around the minimum without settling. Which strategy and priority
is sound?

A) Raise `alpha` over time to bounce harder near the minimum;
larger late steps are said to always find the global optimum
faster, so you should increase the rate as training progresses
without any bound or upper limit.

B) The wander is a bug — just raise `alpha` further at its fixed
value; decay never helps and adaptive methods are orthogonal and
should never be combined with any schedule at all, even jointly.

C) Learning-rate decay replaces adaptive optimizers entirely; you
should choose one or the other and never use both together, since
they are claimed to implement identical per-parameter scaling in
completely different forms.

D) Shrink `alpha` gradually so early steps are large (fast
progress) and late steps are small (settle); decay helps, but a
well-tuned fixed `alpha` and adaptive methods often beat hand-tuned
schedules; step decay needs manual thresholds and is tedious.

### Q20. [LR decay — formulas]
Which decay families are real and which formula is
**inverse-time** decay?

A) Time-based (inverse-time): `alpha=alpha0/(1+decay_rate·epoch)`
(Keras `decay`); step: `alpha=alpha0·drop^{floor(epoch/
epochs_drop)}`; exponential: `alpha0·e^{−k·t}` or `0.95^{epoch}`;
inverse-time drops fast early, then slowly.

B) Only inverse-time exists; step and exponential are just other
names for momentum and RMSProp, not independent learning-rate
schedules at all, and are said to never be used separately ever
in practice.

C) The decay formula is `alpha=alpha0·(1+beta·t)` — the rate is
said to grow linearly with `t`, so training accelerates as
iterations increase, opposite of any real decay schedule now.

D) Learning-rate decay is identical to RMSProp's `s_dW` update;
the formula `s=beta·s+(1−beta)·dW^2` is claimed as the schedule
and global `alpha` is said to never change on its own at all
during training.

---

**Next:** check your picks in `quiz_optimization_answers.md` — each
answer has a 2–4 sentence explanation and the exact
`RESOURCES.md:line` or video section plus the `blog.md:line` the
distractor corrects. If you missed Q4–Q6, revisit
`RESOURCES.md:360-418`; Q10–Q18 rely on `RESOURCES.md:100-113,
130-178, 259-345`.
