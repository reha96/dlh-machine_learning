# Optimization — Resources (intranet project 2293)

Ingested 2026-08-26. Status key: `summary` = full treatment done;
`paywalled` = paywall survived the chain (title + first paragraph only);
`failed-transcript` = captions unavailable, watch manually; `deferred` =
explicitly postponed. References are link-only entries, each verified to
resolve (`curl -sI -L`, HTTP 200) on 2026-08-26.

## Read or watch — Articles

### Hyperparameter (machine learning)

URL: https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)  ·  Date: 2026-07-09 (last edited)  ·  Status: summary

A hyperparameter is any configurable setting fixed before training that shapes
the learning process but is not learned from data: model hyperparameters
(network topology, layer sizes) and algorithm hyperparameters (learning rate,
batch size). The article contrasts them with parameters, explains why they
cannot be learned by gradient methods (the objective is typically not
differentiable in them, and learning them would push straight into
overfitting), and covers tuning, tunability, robustness, and reproducibility.

- Model vs algorithm hyperparameters; ordinary least squares needs none, LASSO adds a regularization one.
- Learning them by gradient descent would overfit (polynomial degree example); tuning is a separate, usually derivative-free optimization.
- Tunability: most performance variation comes from a few hyperparameters — for LSTM, learning rate first, then network size; batch size and momentum barely matter.
- Best mini-batch performance reported with sizes 2–32 (small-batch study), though some advocate thousands.
- Robustness: empirical performance depends on random seeds and implementation details — evaluate over many seeds, especially in RL.
- Reproducibility needs infrastructure (bookkeeping, seed logging); deep learning results are notoriously seed-sensitive.

### Feature scaling

URL: https://en.wikipedia.org/wiki/Feature_scaling  ·  Date: 2025-08-05 (last edited)  ·  Status: summary

Feature scaling normalizes the ranges of independent variables during
preprocessing. Without it, distance-based objectives are dominated by the
widest-range feature, gradient descent converges slowly on elongated cost
surfaces, and regularized losses penalize coefficients unfairly. The page
catalogs the standard scaling methods with their formulas.

- Min-max rescaling: x' = (x − min(x)) / (max(x) − min(x)); general [a,b] variant given.
- Mean normalization: subtract the mean, divide by range (or std).
- Standardization (z-score): x' = (x − mean)/σ → zero mean, unit variance; the default for SVM, logistic regression, neural nets.
- Robust scaling: subtract median, divide by IQR (Q3 − Q1) — outlier-resistant.
- Unit-vector normalization: divide each sample vector by its L1 or L2 norm.

### Why, How and When to Scale your Features

URL: https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e  ·  Date: 2017-12-04 (publication)  ·  Status: summary

Short practitioner piece on feature scaling: raw features differ in magnitude,
units, and range, and algorithms that compute distances silently let the big
ones dominate (5 kg vs 5000 g). It lists four scaling recipes and gives a rule
of thumb for when scaling matters.

- Four methods: standardisation (z-score, μ=0 σ=1), mean normalisation ([−1,1], μ=0), min-max scaling ([0,1]), unit vector.
- Zero-centred variants fit PCA; bounded [0,1] fits image pixel data.
- Scale when the algorithm uses distances or assumes normality: kNN, PCA, gradient descent (θ descends fast on small ranges, slowly on large ones → inefficient oscillation).
- No scaling needed for tree-based models; LDA/Naive Bayes handle ranges by design.

### Normalizing your data (specifically, input and batch normalization)

URL: https://www.jeremyjordan.me/batch-normalization/  ·  Date: 2018-01-26 (publication)  ·  Status: summary

Jordan explains why unnormalized inputs create an awkward loss surface: inputs
on very different scales force the corresponding weights onto different
scales too, so gradient descent must zig-zag with a small learning rate.
Normalizing inputs (zero mean, unit variance, roughly [−1,1]) reshapes the
loss contours rounder; batch norm extends the same idea to hidden-layer
activations so every layer trains on stable inputs.

- Keep inputs roughly in [−1,1] to avoid floating-point precision artifacts and ill-suited default learning rates.
- Batch norm normalizes z^[l] (pre-activation) per batch: z_norm = (z − μ)/√(σ²+ε).
- Learnable rescale/shift: z̃ = γ·z_norm + β lets the network pick whatever distribution suits the next activation (sigmoid suffers if forced to zero-mean).
- A batch-normalized layer needs no bias b — β already shifts the values.
- μ, σ² are computed per batch; γ, β are learned across batches. Effect: orthogonality between layers, deeper networks train without exploding time.

### Moving average

URL: https://en.wikipedia.org/wiki/Moving_average  ·  Date: 2026-06-29 (last edited)  ·  Status: summary

A moving average smooths a series by averaging overlapping windows of it; it
is a convolution, i.e. a low-pass FIR filter. The article walks through the
simple, cumulative, weighted, and exponential variants with update formulas,
plus the moving median as a shock-resistant alternative. This is the
statistics backbone behind exponentially weighted averages in optimizers.

- SMA over window k updates in O(1): SMA_next = SMA_prev + (p_new − p_oldest)/k.
- Cumulative average update: CA_{n+1} = CA_n + (x_{n+1} − CA_n)/(n+1) — no need to store history.
- Weighted MA: linearly decreasing weights n, n−1, …, 1; denominator n(n+1)/2.
- EMA/EWMA: weights decay exponentially, never hitting zero — first-order IIR filter; the form used in momentum/RMSprop/Adam.
- Moving median tolerates outliers better than the mean (optimal for Laplace-distributed noise); SMA is optimal under Gaussian noise.

### An overview of gradient descent optimization algorithms

URL: https://www.ruder.io/optimizing-gradient-descent/  ·  Date: 2016-01-19 (publication)  ·  Status: summary

Ruder's canonical survey of SGD variants and adaptive optimizers. It frames
the three data regimes (batch, stochastic, mini-batch), lists the challenges
(learning-rate choice, schedules fixed in advance, one rate for all
parameters, saddle points), then derives each optimizer's update rule as a fix
for the previous one's flaw, ending with practical advice: use adaptive
methods for sparse data and fast convergence; Adam adds bias correction and
momentum to RMSprop.

Crucial formulas:

- Momentum: v_t = γ·v_{t−1} + η∇θJ(θ); θ ← θ − v_t, with γ ≈ 0.9.
- Nesterov: v_t = γ·v_{t−1} + η∇θJ(θ − γv_{t−1}); look-ahead before grading.
- Adagrad: θ_{t+1,i} = θ_{t,i} − η/√(G_t,ii + ε) · g_t,i — G accumulates squared gradients; monotone shrinkage kills learning late in training.
- Adadelta/RMSprop replace the sum by a decaying average E[g²]_t = γE[g²]_{t−1} + (1−γ)g²_t; RMSprop uses 0.9/0.1 split and divides: θ ← θ − η/√(E[g²]_t + ε)·g_t.
- Adam keeps both moments: m_t = β₁m_{t−1} + (1−β₁)g_t, v_t = β₂v_{t−1} + (1−β₂)g²_t, bias-corrected m̂_t = m_t/(1−β₁^t), v̂_t = v_t/(1−β₂^t); update θ ← θ − η·m̂_t/(√v̂_t + ε); defaults β₁=0.9, β₂=0.999, ε=1e−8.
- LR-decay appears as schedule-based annealing (reduce η by pre-defined schedule or when epoch-to-epoch change stalls) — the fix SGD needs since it overshoots at constant η.
- Extra strategies: shuffle each epoch, batch norm re-establishes normalization per mini-batch, early stopping ("beautiful free lunch"), gradient noise N(0, σ²_t).

### A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size

URL: https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/  ·  Date: 2019-08-19 (publication)  ·  Status: summary

Brownlee contrasts the three flavors of gradient descent by how many examples
drive each update, then argues mini-batch is the default and shows how to size
it. Batch GD is stable but slow and memory-hungry; SGD is fast to learn but
noisy and loses vectorization; mini-batch balances both.

- Batch GD: error over all examples, one update per epoch. SGD: update per example (online learning).
- Mini-batch upsides: more frequent updates than batch (robust convergence, escapes local minima), cheaper than per-example updates, dataset needn't fit in memory.
- Batch sizes tuned to hardware powers of two: 32, 64, 128, 256.
- Tip 1: batch_size = 32 is a good default (Bengio 2012); the 2018 small-batch study found m ≤ 32, often 2–4, best for stability and generalization.
- Tips 2–3: tune batch size from validation-error-vs-time curves; tune batch size and learning rate last.

### Stochastic Gradient Descent with momentum

URL: https://medium.com/data-science/stochastic-gradient-descent-with-momentum-a84097641a5d  ·  Date: 2017-12-04 (publication)  ·  Status: summary

Bushaev builds momentum from exponentially weighted averages: mini-batch
gradients are noisy estimates, so averaging recent gradients denoises the
update direction and accelerates convergence, especially through ravines near
local minima where plain SGD oscillates wall to wall.

- EWMA: V_t = β·V_{t−1} + (1−β)·θ_t; β=0.9 ≈ averaging over ~1/(1−β)=10 points.
- Weights decay geometrically (β^i·(1−β)); once below 1/e older terms are effectively forgotten.
- Bias-corrected variant V_t/(1−β^t) fixes the low start; usually skipped for momentum since learning stabilizes fast.
- Andrew Ng form: v_dW = β·v_dW + (1−β)·dW; W ← W − α·v_dW. Literature often drops the (1−β) factor and rescales α instead.
- Nesterov: grade at θ − β·v_{t−1} (look-ahead point) instead of θ.
- Two reasons momentum wins: better estimate of the true gradient than one noisy mini-batch; damped oscillation across steep ravine walls.

### Understanding RMSprop — faster neural network learning

URL: https://medium.com/data-science/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a  ·  Date: 2018-09-02 (publication)  ·  Status: summary

Bushaev derives RMSprop twice over: as rprop adapted to mini-batches, and as
Adagrad with its runaway denominator cured. Rprop uses only gradient signs
with per-weight step sizes — great for full-batch, broken for mini-batches
because sign steps don't average. RMSprop keeps the moving average of squared
gradients and divides by its root, so each weight gets its own scale.

- Core update: s_dW = β·s_dW + (1−β)·dW²; W ← W − α·dW/√s_dW; β default 0.9.
- "RMS" = root mean square: square the derivatives, average, take the root at division time.
- Adagrad's running sum grows monotonically → learning rate collapses toward zero (bad in non-convex settings/saddle points); the moving average forgets old terms.
- Directions with large oscillating gradients get large s values → divided down; small-gradient directions get relatively larger effective steps.
- Unpublished: proposed by Geoff Hinton in Coursera lecture 6e, spread through the course rather than a paper; second in popularity only to Adam.

### Adam — latest trends in deep learning optimization

URL: https://medium.com/data-science/adam-latest-trends-in-deep-learning-optimization-6be9a291375c  ·  Date: 2018-10-22 (publication)  ·  Status: summary

Long treatment of Adam = RMSprop + SGD-with-momentum: first-moment average m
(momentum part, β₁=0.9), second-moment average v of squared gradients
(RMSprop part, β₂=0.999), both bias-corrected because zero initialization
biases them toward zero early. Bushaev then surveys why Adam sometimes
generalizes worse than SGD+momentum and the fixes: AMSGrad, SWATS,
decoupled weight decay (AdamW), ND-Adam.

- Update: m_t = β₁m_{t−1} + (1−β₁)g_t; v_t = β₂v_{t−1} + (1−β₂)g²_t; m̂ = m_t/(1−β₁^t); v̂ = v_t/(1−β₂^t); w ← w − η·m̂/(√v̂ + ε).
- Bias correction exists because E[m_t] = (1−β₁^t)·E[g] under zero init; divide it out.
- Step size is bounded by η and invariant to gradient magnitude — helps cross plateaus and saddle points.
- Reddi et al.: exponential averaging forgets rare informative gradients → non-convergence counterexample; AMSGrad keeps max of past v_t (little practical gain in later tests).
- Loshchilov & Hutter: L2 regularization ≠ weight decay for Adam; decoupled weight decay (AdamW) also decouples lr from regularization tuning.
- Wilson et al.: adaptive methods generalize worse than SGD+momentum on many tasks; SWATS switches Adam→SGD mid-training.

### Learning Rate Schedules and Adaptive Learning Rate Methods for Deep Learning

URL: https://medium.com/data-science/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1  ·  Date: 2017-07-29 (publication)  ·  Status: summary

Suki Lau benchmarks learning-rate schedules against adaptive optimizers on a
CIFAR-10 CNN in Keras. Schedules shrink a single global learning rate on a
timer; adaptive methods give each parameter its own effective rate based on
gradient history and need far less tuning.

- Constant lr baseline: SGD(lr=0.1, momentum=0, decay=0).
- Time-based decay: lr = lr0/(1 + k·t) (Keras `decay` argument applies per iteration).
- Step decay: lr = lr0 · drop^floor(epoch/epochs_drop), e.g. halve every 10 epochs, via LearningRateScheduler callback.
- Exponential decay: lr = lr0·e^(−k·t); any custom schedule plugs into the same callback.
- Adaptive roster and Keras defaults: Adagrad(0.01), Adadelta(lr=1.0, rho=0.95), RMSprop(0.001, rho=0.9), Adam(0.001, β₁=0.9, β₂=0.999).
- In her experiments adaptive methods beat schedules; Adadelta scored best among the adaptives.

### The Feynman Learning Technique

URL: https://fs.blog/feynman-learning-technique/  ·  Date: 2021-02-22 (publication)  ·  Status: summary

Farnam Street's write-up of Feynman's study method: you only understand what
you can explain simply. Teaching a concept to a (pretend) sixth-grader exposes
jargon that masks gaps; going back to sources fills them; organizing and
transmitting locks the knowledge in. Knowing the name of something is not
knowing the thing.

- Step 1: write the concept out as if teaching a child — nowhere to hide behind vocabulary.
- Step 2: note where the explanation breaks; return to source material until simple language suffices.
- Step 3: organize into a narrative, read aloud, iterate.
- Step 4 (optional): transmit to someone unfamiliar; their questions finish the learning.
- Jargon is the classic mask for non-understanding; simple terms can be recombined and reused, memorized labels cannot.

## Read or watch — YouTube (deeplearning.ai Course 2)

### Normalizing Inputs

URL: https://www.youtube.com/watch?v=FDCfw-YqWTE  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Ng shows input normalization as two steps: subtract the mean μ (computed over
the m training features), then divide by the per-feature variance σ², giving
every feature zero mean and unit variance. Unnormalized features on wildly
different scales (1–1000 vs 0–1) produce an elongated cost bowl whose
contours force tiny learning rates and slow zig-zag descent; equal scales make
the surface round so descent heads nearly straight to the minimum.

- Use the SAME μ and σ² from the training set to normalize test data — never re-estimate on test.
- Cost J becomes elongated when feature ranges differ because w₁, w₂ then live on different scales.
- Similar ranges (0–1, −1–1, 1–2) are fine; dramatic mismatches hurt.
- Normalization rarely harms — do it when unsure.

### Mini Batch Gradient Descent

URL: https://www.youtube.com/watch?v=4qJaSmvhxi8  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Vectorized batch gradient descent still waits for all m examples (say 5
million) before one step. Ng splits the set into mini-batches of 1000:
X^{t}, Y^{t}, t = 1…5000, and runs one full forward/backward/update cycle per
mini-batch using vectorization within the batch.

- Notation: {t} indexes mini-batches, alongside (i) for examples and [l] for layers; X^{t} is n_x×1000.
- One pass over all mini-batches = one epoch; batch GD takes 1 step per epoch, mini-batch takes 5000.
- Per batch: forward prop on X^{t}, cost J^{t} = (1/1000)Σ loss (+ regularizer), backprop, update W, b.
- Fast optimization algorithms matter because applied ML is empirical — teams iterate over many models.

### Understanding Mini-Batch Gradient Descent

URL: https://www.youtube.com/watch?v=-_4Zi8fCZO4  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

The mini-batch cost curve trends downward but is noisy — each batch is a
different "dataset", some easy, some hard. Size extremes: size m recovers
batch GD (too slow per step on big data), size 1 is stochastic GD (noisy,
loses vectorization). Something in between gets both vectorization speed and
progress before the epoch ends.

- Batch GD cost must decrease every iteration or the learning rate is too big; mini-batch cost only trends down.
- Guidelines: training set ≤ 2000 → just use batch GD; else typical sizes 64–512, preferably powers of 2 (1024 works too).
- Keep X^{t}, Y^{t} small enough to fit CPU/GPU memory — overflow craters performance.
- Batch size is another hyperparameter; try a few powers of two.

### Exponentially Weighted Averages

URL: https://www.youtube.com/watch?v=lAq96T8FkTw  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Using London temperatures, Ng introduces V_t = β·V_{t−1} + (1−β)·θ_t, a
one-number running average that smooths noisy sequences. β≈0.9 averages
roughly the last 10 days; larger β (0.98) is smoother but lags further right;
smaller β (0.5) tracks closely but stays jittery.

- Formula generalized from the 0.9·prev + 0.1·today pattern.
- Window intuition: averages about 1/(1−β) days.
- Trade-off: bigger window = smoother but slower to adapt; smaller = responsive but noisy.
- This recursion is the building block of momentum, RMSprop, and Adam.

### Understanding Exponentially Weighted Averages

URL: https://www.youtube.com/watch?v=NxTFlzBjS-4  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Ng expands the recursion algebraically: V_100 is a sum of θ_t terms weighted
by an exponentially decaying sequence 0.1, 0.1·0.9, 0.1·0.9², … — an
element-wise product of the data with a decay kernel. Since 0.9^10 ≈ 1/e ≈
0.35, weights become negligible after ~1/(1−β) steps, justifying the window
rule of thumb.

- All coefficients sum to (about) one — it is a true weighted average up to bias correction.
- (1−ε)^(1/ε) ≈ 1/e is the identity behind the 1/(1−β) window heuristic.
- Implementation: keep ONE real number v_θ, overwrite each step: v_θ := β·v_θ + (1−β)·θ_t.
- Cheaper in memory/compute than explicit window sums; slightly less accurate — the efficiency is why ML uses it.

### Bias Correction of Exponentially Weighted Averages

URL: https://www.youtube.com/watch?v=lWzo8CajF5s  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

With v₀ = 0 the early estimates start far too low (day 1 gives v₁ = 0.02·θ₁).
Dividing by (1−β^t) renormalizes: at t=2, v₂/(1−0.98²) becomes an exact
weighted average of θ₁, θ₂. As t grows, β^t → 0 and the correction vanishes —
it matters only during warm-up.

- Purple (uncorrected) vs green (corrected) curve: correction lifts the initial segment.
- In practice most implementations of momentum skip bias correction and tolerate the brief bias.
- Adam does use it (per-parameter, both moments).

### Gradient Descent With Momentum

URL: https://www.youtube.com/watch?v=k8fTYJPd3_I  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Momentum replaces raw gradients in the update by their exponentially weighted
averages: v_dW = β·v_dW + (1−β)·dW, then W ← W − α·v_dW (same for b). Vertical
oscillations average toward zero while horizontal progress accumulates, so a
larger learning rate becomes safe. Ball-downhill analogy: derivatives give
acceleration, β acts as friction.

- Works identically for batch and mini-batch GD.
- β = 0.9 is the robust default (≈ averaging last 10 gradients).
- Bias correction usually skipped — the average warms up within ~10 iterations.
- Initialize v_dW, v_db as zeros with same shape as dW, db.
- Some texts drop the (1−β) factor (v_dW = β·v_dW + dW); equivalent up to rescaling α, but then β changes also rescale the update — Ng prefers keeping (1−β).

### RMSProp

URL: https://www.youtube.com/watch?v=_e-LFe_igno  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

RMSprop dampens oscillations per-dimension: keep s_dW = β·s_dW +
(1−β)·dW² (element-wise square) and update W ← W − α·dW/√(s_dW). Steep
directions accumulate large squared-gradient averages, shrinking their steps;
flat directions keep moving. Result: larger learning rates converge without
diverging vertically.

- Named root-mean-square-prop: square, average, divide by the root.
- Uses β₂ notation to distinguish from momentum's β when combined later.
- Add small ε (≈1e−8) inside the root for numerical stability.
- Fun fact: published only as Hinton's Coursera lecture 6e, never a paper.

### Adam Optimization Algorithm

URL: https://www.youtube.com/watch?v=JXQT_vxqwIs  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Adam = momentum + RMSprop with bias correction: v from β₁ (first moment),
s from β₂ (second moment), both corrected by dividing by (1−β^t), update
W ← W − α·v_corrected/(√s_corrected + ε). One of the rare algorithms proven
robust across architectures.

- Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e−8 (ε barely matters; nobody tunes it).
- Tune α; β₁, β₂ rarely touched.
- Name: Adaptive Moment Estimation — mean of derivatives (first moment) and average of squares (second moment).
- Rare case of an optimizer that stood up to wide testing; Ng recommends trying it.

### Learning Rate Decay

URL: https://www.youtube.com/watch?v=QzulmoOg2JE  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

With a fixed α, mini-batch noise keeps the iterate wandering near but never
into the minimum. Slowly reducing α gives big early steps and small final
oscillations. Primary formula: α = α₀ / (1 + decay_rate × epoch_num), with
decay_rate another hyperparameter (example: α₀=0.2, decay 1 → 0.1, 0.67, 0.5,
0.4 per epoch).

- Alternatives: exponential decay 0.95^epoch·α₀; α = k/√epoch·α₀; α = k/√mini-batch-t·α₀; discrete staircase halving.
- Manual gradual decay works for single long runs.
- Priority check: a well-chosen fixed α matters more; decay helps but sits lower on the tuning list.

### Normalizing Activations in a Network

URL: https://www.youtube.com/watch?v=tNIpEZLv_eg  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Batch norm (Ioffe & Szegedy) extends input normalization to hidden layers,
making hyperparameter search easier and very deep networks trainable. It
normalizes z-values (not a; pre-activation is the common default) per layer
per batch: μ, σ² over the batch, z_norm = (z−μ)/√(σ²+ε).

- Learned rescale: z̃ = γ·z_norm + β with γ, β updated like weights — the network picks its own mean/variance instead of being forced to 0/1.
- If γ = √(σ²+ε) and β = μ the transform is exactly invertible (identity) — the family spans normalized and unnormalized.
- Motivation: normalize inputs helped layer 1; later layers deserve the same stability for their "inputs" (earlier activations).
- Forcing sigmoid inputs to zero-mean wastes the non-linear tail — hence learnable shift.

### Fitting Batch Norm Into Neural Networks

URL: https://www.youtube.com/watch?v=em6dfRxYkYU  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Placement: compute z^[l], apply BN (parameters β^[l], γ^[l]) to get z̃^[l],
then activation a^[l] = g(z̃^[l]); repeat per layer, all within each
mini-batch. Because the batch mean-subtraction cancels any constant, the
layer bias b^[l] is useless and removed — β^[l] plays the bias role. Train
β, γ with any optimizer; frameworks do this in one line (tf.nn.batch_normalization).

- BN's β has nothing to do with momentum's β hyperparameter — name collision only.
- Per mini-batch: μ, σ² estimated on just that batch's z values.
- Parameter dimensions: β^[l], γ^[l] are (n^[l], 1), one pair per hidden unit.
- Full loop: for each mini-batch — forward prop with BN, backprop dW, dβ, dγ, update with GD/momentum/RMSprop/Adam.

### Why Does Batch Norm Work?

URL: https://www.youtube.com/watch?v=nUUqwaxLnWs  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Three stacked intuitions: (1) same benefit as normalizing inputs, applied to
hidden units. (2) Covariate-shift analogy — black-cat classifier fails on
colored cats; deep layers see shifting input distributions as earlier layers
learn. BN pins each layer's z-mean/variance, weakening the coupling between
layers so each learns more independently. (3) Per-batch μ, σ² are noisy, which
adds slight multiplicative/additive noise to activations — a dropout-like,
minor regularization side effect.

- Bigger mini-batch (512 vs 64) reduces that noise and thus the regularization effect — a quirk worth knowing.
- Don't use BN as your regularizer; it's incidental and small.
- Main takeaway: later layers get firmer ground; earlier-layer updates shift activations less.

### Batch Norm At Test Time

URL: https://www.youtube.com/watch?v=5qefnAek8OA  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

At test time you may have one example — no batch to average μ, σ² over. The
standard trick: during training, track exponentially weighted (running)
averages of each layer's μ and σ² across mini-batches; at inference use those
stored estimates inside the same normalize-rescale equations with the learned
γ, β.

- Any reasonable estimate of μ, σ² works; the method is robust.
- Frameworks ship defaults for the running estimates.
- Contrast to remember: training μ/σ² come from the current batch; test μ/σ² come from the running averages collected while training.

### The Problem of Local Optima

URL: https://www.youtube.com/watch?v=fODpu1-lNTw  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary

Low-dimensional pictures mislead: in a 20,000-dimensional parameter space, a
local optimum requires the gradient to vanish with upward curvature in every
direction — astronomically unlikely (~2^-20000). Nearly all zero-gradient
points are saddle points, where some directions bend up and others down. The
real hazard is plateaus — long flat stretches of near-zero gradient that slow
training.

- Saddle-point picture: curves up one way, down the other (the horse-saddle name).
- Plateaus waste time; momentum, RMSprop, Adam help escape them faster.
- Lesson: 2D/3D intuitions about loss surfaces mostly fail at high dimension.

## References

Link-only entries; verified resolving via `curl -sI -L` on 2026-08-26.
Last-Modified dates recorded where present.

- R1. [numpy.random.permutation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html) — verified HTTP 200 · Date: 2026-06-28 (page last modified) · Status: verified
- R2. [tf.nn.moments](https://www.tensorflow.org/api_docs/python/tf/nn/moments) — verified HTTP 200 · Date: 2024-04-26 (page last modified) · Status: verified
- R3. [tf.keras.optimizers.SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) — verified HTTP 200 · Date: 2024-06-07 (page last modified) · Status: verified
- R4. [tf.keras.optimizers.RMSprop](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop) — verified HTTP 200 · Date: 2024-06-07 (page last modified) · Status: verified
- R5. [tf.keras.optimizers.Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) — verified HTTP 200 · Date: 2024-06-07 (page last modified) · Status: verified
- R6. [tf.nn.batch_normalization (r2.6)](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/batch_normalization) — verified HTTP 200 · Date: 2021-08-16 (page last modified) · Status: verified
- R7. [tf.keras.optimizers.schedules.InverseTimeDecay](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay) — verified HTTP 200 · Date: 2024-06-07 (page last modified) · Status: verified
- R8. [deeplearning.ai](https://www.deeplearning.ai/) — verified HTTP 200 · Date: unknown (no Last-Modified header) · Status: verified

## Quiz Hooks

- Hyperparameter — setting fixed before training (learning rate, batch size, layers); not learned from data.
- Parameter — value the model learns from data (weights W, biases b).
- Feature scaling — bringing all input features onto comparable ranges so no feature dominates distances or gradients.
- Standardization (z-score) — x' = (x − μ)/σ: zero mean, unit variance.
- Min-max scaling — x' = (x − min)/(max − min): squeezes into [0, 1].
- Mean normalization — x' = (x − μ)/(max − min): range [−1, 1].
- Input normalization effect — rounds the cost contours so gradient descent can take bigger, straighter steps.
- Same μ, σ² rule — normalize test data with statistics computed on training data only.
- Mini-batch gradient descent — one update per small batch; vectorizes within batches, makes progress before the epoch ends.
- Epoch — one full pass over the training set.
- Batch size guideline — ≤2000 examples: batch GD fine; otherwise 64–512, power of 2, fits GPU memory.
- Exponentially weighted average — V_t = β·V_{t−1} + (1−β)·θ_t; one-number smoothed running average.
- 1/(1−β) rule — approximate number of past samples an EWMA effectively averages.
- Bias correction — divide V_t by (1−β^t) to undo the zero-start underestimate; matters only early.
- Momentum — v_dW = β·v_dW + (1−β)·dW; update with v_dW to damp oscillation, β ≈ 0.9.
- RMSprop — s_dW = β₂·s_dW + (1−β₂)·dW²; divide update by √(s_dW + ε) to equalize step sizes per direction.
- Adam — momentum + RMSprop + bias correction; update α·v_corr/(√s_corr + ε).
- Adam defaults — β₁ = 0.9, β₂ = 0.999, ε = 1e−8; tune only α.
- Learning rate decay — α = α₀/(1 + decay_rate·epoch): big steps early, tight oscillation near the minimum.
- Time-based decay formula — lr = lr0/(1 + k·t).
- Step decay formula — lr = lr0·drop^floor(epoch/epochs_drop).
- Exponential decay formula — lr = lr0·e^(−kt).
- Batch norm placement — after computing z^[l], before activation: z̃ = γ·z_norm + β.
- Batch norm removes bias — mean subtraction cancels b^[l]; learned β^[l] replaces it.
- Batch norm at test time — use running (exponentially weighted) averages of μ and σ² stored during training.
- Covariate shift (batch norm sense) — earlier layers changing distributions destabilize later layers; BN pins each layer's mean/variance.
- Saddle point — zero-gradient point curving up in some directions, down in others; the typical stationary point in high dimensions.
- Local optima myth — bad local optima are rare in high-dimensional spaces; plateaus (slow learning) are the real problem.
- Moving average — average over a sliding window; a low-pass filter for noisy series.
- Tunability — measure of how much performance tuning one hyperparameter can buy (learning rate usually highest).
