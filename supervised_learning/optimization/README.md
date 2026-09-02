# Optimization — Supervised Learning

Optimization toolkit from feature scaling and batch normalization through mini-batch, exponentially weighted averages, momentum, RMSProp, Adam and learning-rate decay (intranet project 2293).

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Distinguish hyperparameters (learning rate, batch size, depth) from learned parameters (W, b) and describe tunability and tuning practice |
| 2 | Apply feature scaling — standardization `x'=(x−mean)/std` and min-max `x'=(x−min)/(max−min)` — to round cost contours and equalize feature influence |
| 3 | Explain why input normalization uses train-only `mean, std` on test data and how skewed features (log transform) interact with scaling |
| 4 | Implement batch normalization: per-mini-batch `mu, sigma2`, `z_norm=(z−mu)/sqrt(sigma2+eps)`, learned `gamma, beta`, and running averages at test time |
| 5 | Construct mini-batches: shuffle with `np.random.permutation`, slice with remainder handling, size guidelines (32–512 power of 2, `m≤2000→batch GD`) |
| 6 | Compute exponentially weighted averages `V_t=beta·V_{t-1}+(1−beta)·theta_t`, window `≈1/(1−beta)`, and bias correction `V_t/(1−beta^t)` |
| 7 | Implement momentum: first-moment `v=beta·v+(1−beta)·dW`, `W←W−alpha·v`, ravine damping and larger safe `alpha` |
| 8 | Implement RMSProp: second-moment `s=beta·s+(1−beta)·dW²`, `W←W−alpha·dW/sqrt(s+eps)`, per-parameter adaptive rates |
| 9 | Implement Adam: both moments `m,v` with bias correction `m/(1−beta1^t), v/(1−beta2^t)`, `W←W−alpha·m_hat/(sqrt(v_hat)+eps)`, defaults `beta1=0.9, beta2=0.999, eps=1e-8` |
| 10 | Apply learning-rate decay schedules — inverse-time `alpha0/(1+decay·epoch)`, step `alpha0·drop^{floor(epoch/epochs_drop)}`, exponential `alpha0·e^{−k·t}` — and trade-offs vs adaptive optimizers |
| 11 | Discuss high-dimensional loss surfaces: saddle points vs plateaus, why bad local optima are rare and momentum/adaptive methods help |

---

## Task-by-Task Reference

### Task 0 — Normalization Constants (`0-norm_constants.py`)

**Challenge:** Obtain the constants needed to standardize a dataset without yet transforming it.

**Approach:** Compute per-feature statistics over the `m` rows: mean and standard deviation with `np.mean(X, axis=0)` and `np.std(X, axis=0)`. Returned as `(mean, std)` shape `(nx,)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.mean(X, axis=0)` / `np.std(X, axis=0)` | Per-feature population statistics for standardization |

> **Key takeaway:** Standardization constants are population moments over the training set; they are reused everywhere else.

---

### Task 1 — Normalize (`1-normalize.py`)

**Challenge:** Apply standardization given precomputed constants, handling broadcasting.

**Approach:** Return `(X − m)/s` with numpy broadcasting; `m, s` shape `(nx,)` broadcast over `(d, nx)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `(X − m)/s` broadcasting | Zero-mean, unit-variance transform `x'=(x−mean)/std` |

> **Key takeaway:** Normalization is an affine map; the same train `mean, std` must normalize test data.

---

### Task 2 — Shuffle Data (`2-shuffle_data.py`)

**Challenge:** Permute examples identically across `X` and `Y` before batching.

**Approach:** Build index permutation `perm = np.random.permutation(m)` and index both arrays: `X[perm], Y[perm]`. Relies on `m = X.shape[0]`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.random.permutation(m)` | Uniform random permutation of `0…m−1` for shuffling without replacement |

> **Key takeaway:** Shuffling preserves example-label correspondence and breaks order bias before mini-batching.

---

### Task 3 — Mini-Batch (`3-mini_batch.py`)

**Challenge:** Split a shuffled dataset into mini-batches, allowing a smaller final batch.

**Approach:** Call `shuffle_data`, then slice with `range(0, m, batch_size)` and collect `(X[i:i+batch_size], Y[i:i+batch_size])` tuples.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `range(0, m, batch_size)` slicing | Partition into batches; last slice naturally smaller if `m % batch_size ≠ 0` |
| `shuffle_data` composition | Shuffle-then-slice pattern for stochastic optimization |

> **Key takeaway:** Mini-batch GD makes `m/batch_size` updates per epoch, vectorized inside each batch and more frequent than batch GD.

---

### Task 4 — Moving Average (`4-moving_average.py`)

**Challenge:** Smooth a noisy sequence with an exponentially weighted average that is unbiased early.

**Approach:** Iterate `v = beta·v + (1−beta)·d` and correct `v/(1−beta**(i+1))`; accumulate bias-corrected values.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| EWMA `V_t=beta·V_{t-1}+(1−beta)·theta_t` | One-scalar running average; window `≈1/(1−beta)` |
| Bias correction `V_t/(1−beta^t)` | Undo zero-initialization underestimate during warm-up |

> **Key takeaway:** EWMA is the shared primitive behind momentum, RMSProp and Adam; bias correction matters only for the first `≈1/(1−beta)` steps.

---

### Task 5 — Momentum (numpy) (`5-momentum.py`)

**Challenge:** Damp ravine oscillations by averaging recent gradients.

**Approach:** Update first moment `v = beta1·v + (1−beta1)·grad`, then `var = var − alpha·v`. No bias correction (warm-up tolerated).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `v = beta1·v + (1−beta1)·grad; var −= alpha·v` | Momentum update; `beta1=0.9` default damps vertical oscillation, accumulates horizontal progress |

> **Key takeaway:** Momentum replaces raw `dW` with its EWMA, allowing a larger `alpha` on elongated bowls.

---

### Task 6 — Momentum (TensorFlow) (`6-momentum.py`)

**Challenge:** Wrap the numpy momentum rule as a TensorFlow optimizer.

**Approach:** Return `tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `tf.keras.optimizers.SGD` with `momentum` | Framework equivalent of the numpy momentum update |

> **Key takeaway:** In TF, momentum is a parameter of SGD; `alpha` and `beta1` map directly to the numpy implementation.

---

### Task 7 — RMSProp (numpy) (`7-RMSProp.py`)

**Challenge:** Give each parameter its own effective learning rate based on recent squared-gradient magnitude.

**Approach:** Track second moment `s = beta2·s + (1−beta2)·grad²` (element-wise square), update `var −= alpha·grad/(sqrt(s)+epsilon)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `s = beta2·s+(1−beta2)·dW²; var −= alpha·dW/sqrt(s+eps)` | Per-dimension adaptive rate; large oscillating directions get smaller steps |

> **Key takeaway:** RMSProp cures Adagrad's monotonically growing denominator by decaying the squared-gradient average.

---

### Task 8 — RMSProp (TensorFlow) (`8-RMSProp.py`)

**Challenge:** Expose the same RMSProp rule through TensorFlow.

**Approach:** Return `tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)`; note TF names the decay `rho`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `tf.keras.optimizers.RMSprop` with `rho` | TF's RMSProp; `rho` ↔ `beta2`, `epsilon` for stability |

> **Key takeaway:** TF's `rho` is the EWMA decay for `s`; the numpy formula maps one-to-one.

---

### Task 9 — Adam (numpy) (`9-Adam.py`)

**Challenge:** Combine momentum and RMSProp with correction for zero initialization.

**Approach:** Update `v = beta1·v+(1−beta1)·grad`, `s = beta2·s+(1−beta2)·grad²`, bias-correct `v/(1−beta1**t)`, `s/(1−beta2**t)`, step `var −= alpha·v_corr/(sqrt(s_corr)+epsilon)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Two EWMAs `m` (first moment) + `v` (second moment) + dual bias correction | Adam = momentum + RMSProp; bounded step `≤alpha`, invariant to gradient rescaling |

> **Key takeaway:** Adam's bias correction is essential because both moments start at zero; after `t≫1/(1−beta)` it vanishes.

---

### Task 10 — Adam (TensorFlow) (`10-Adam.py`)

**Challenge:** Provide Adam as a TF optimizer.

**Approach:** Return `tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `tf.keras.optimizers.Adam` with `beta_1, beta_2` | Framework Adam; maps directly to numpy `beta1, beta2, epsilon` |

> **Key takeaway:** Defaults `beta1=0.9, beta2=0.999, eps=1e-8` are robust; only `alpha` is routinely tuned.

---

### Task 11 — Learning Rate Decay (numpy) (`11-learning_rate_decay.py`)

**Challenge:** Shrink `alpha` on a clock, not on gradients, for a staircase inverse-time schedule.

**Approach:** Return `alpha / (1 + decay_rate * (global_step // decay_step))`; `//` gives `floor(t/decay_step)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Inverse-time staircase `alpha/(1+decay·floor(step/decay_step))` | Big early steps, small late steps; power=1 schedule |

> **Key takeaway:** Decay is a global, time-based scalar schedule — fundamentally different from gradient-based adaptivity.

---

### Task 12 — Learning Rate Decay (TensorFlow) (`12-learning_rate_decay.py`)

**Challenge:** Create the same schedule as a TF learning-rate schedule object.

**Approach:** Return `tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=alpha, decay_steps=decay_step, decay_rate=decay_rate, staircase=True)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `tf.keras.optimizers.schedules.InverseTimeDecay` with `staircase=True` | TF schedule equivalent; `staircase=True` matches the `//` floor behavior |

> **Key takeaway:** Schedules plug into optimizers; `staircase=True` is the discrete halving variant, `False` is continuous `1/(1+decay·step/decay_steps)`.

---

### Task 13 — Batch Norm (numpy) (`13-batch_norm.py`)

**Challenge:** Normalize unactivated outputs per mini-batch, then learn an affine repair.

**Approach:** Compute `mean = np.mean(Z, axis=0)`, `var = np.var(Z, axis=0)`, `Z_std = (Z−mean)/sqrt(var+epsilon)`, output `Z_std·gamma + beta` where `gamma, beta` shape `(1, n)` broadcast.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.mean/var` over `axis=0` + `gamma·z_norm+beta` | Per-unit batch statistics and learnable scale/shift; `epsilon` inside sqrt |

> **Key takeaway:** Batch norm standardizes `Z` per column (per hidden unit) on the current mini-batch; `gamma, beta` let the net undo normalization if needed.

---

### Task 14 — Batch Norm (TensorFlow) (`14-batch_norm.py`)

**Challenge:** Insert batch norm into a layer before its activation, without a bias term.

**Approach:** Dense without activation (`VarianceScaling(mode='fan_avg')`), then `mean,var = tf.nn.moments(Z, axes=[0])`, learnable `gamma=tf.Variable(ones((1,n)))`, `beta=tf.Variable(zeros((1,n)))`, `Z_norm=tf.nn.batch_normalization(Z, mean, var, beta, gamma, variance_epsilon=1e-7)`, return `activation(Z_norm)`. Bias `b` is omitted — `beta` replaces it.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `tf.nn.moments` + `tf.nn.batch_normalization` + `VarianceScaling(fan_avg)` | Low-level BN before activation; fan-average init keeps `Var(Z)≈1` forward/backward |

> **Key takeaway:** Dense `b` cancels under mean subtraction, so BN layers have no bias; `beta` is the effective bias and BN is applied per mini-batch every training step, with running averages used at test time.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `np.mean/std` per feature `axis=0` for constants | Preprocessing / Statistics |
| 1 | `(X−m)/s` standardization via broadcasting | Preprocessing / Feature scaling |
| 2 | `np.random.permutation` shuffling of `X,Y` | Data handling / Stochasticity |
| 3 | Shuffle-then-slice mini-batch creation with remainder | Optimization / Mini-batch GD |
| 4 | EWMA `V_t=beta·V_{t-1}+(1−beta)·theta` + bias correction `/(1−beta^t)` | Smoothing / Moments |
| 5 | Momentum `v=beta·v+(1−beta)·dW; W−=alpha·v` | Optimization / Momentum |
| 6 | `tf.keras.optimizers.SGD(momentum=)` | TF Optimizer API |
| 7 | RMSProp `s=beta·s+(1−beta)·dW²; W−=alpha·dW/sqrt(s+eps)` | Optimization / Adaptive rates |
| 8 | `tf.keras.optimizers.RMSprop(rho=)` | TF Optimizer API |
| 9 | Adam two moments + dual bias correction, `W−=alpha·m_hat/(sqrt(v_hat)+eps)` | Optimization / Adam |
| 10 | `tf.keras.optimizers.Adam(beta_1,beta_2)` | TF Optimizer API |
| 11 | Inverse-time staircase `alpha/(1+decay·floor(step/decay_step))` | Schedules / LR decay |
| 12 | `tf.keras.optimizers.schedules.InverseTimeDecay(staircase=True)` | TF Schedule API |
| 13 | Per-batch `mean/var axis=0`, `gamma·z_norm+beta`, `eps` inside sqrt | Normalization / Batch norm |
| 14 | `tf.nn.moments` + `tf.nn.batch_normalization` + `VarianceScaling(fan_avg)` before activation, no bias | TF Layers / Batch norm |

---

## Resources

**Articles (intranet Read or watch):**

- [Hyperparameter (machine learning) — Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
- [Feature scaling — Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)
- [Why, How and When to Scale your Features — GreyAtom](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e)
- [Normalizing your data (input and batch normalization) — Jeremy Jordan](https://www.jeremyjordan.me/batch-normalization/)
- [Moving average — Wikipedia](https://en.wikipedia.org/wiki/Moving_average)
- [An overview of gradient descent optimization algorithms — Ruder](https://www.ruder.io/optimizing-gradient-descent/)
- [A Gentle Introduction to Mini-Batch Gradient Descent — Brownlee](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
- [Stochastic Gradient Descent with momentum — Bushaev](https://medium.com/data-science/stochastic-gradient-descent-with-momentum-a84097641a5d)
- [Understanding RMSprop — Bushaev](https://medium.com/data-science/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a)
- [Adam — latest trends in deep learning optimization — Bushaev](https://medium.com/data-science/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
- [Learning Rate Schedules and Adaptive Learning Rate Methods — Lau](https://medium.com/data-science/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)
- [The Feynman Learning Technique — Farnam Street](https://fs.blog/feynman-learning-technique/)

**YouTube — deeplearning.ai Course 2 (Andrew Ng):**

- [Normalizing Inputs](https://www.youtube.com/watch?v=FDCfw-YqWTE)
- [Mini-Batch Gradient Descent](https://www.youtube.com/watch?v=4qJaSmvhxi8)
- [Understanding Mini-Batch Gradient Descent](https://www.youtube.com/watch?v=-_4Zi8fCZO4)
- [Exponentially Weighted Averages](https://www.youtube.com/watch?v=lAq96T8FkTw)
- [Understanding Exponentially Weighted Averages](https://www.youtube.com/watch?v=NxTFlzBjS-4)
- [Bias Correction of Exponentially Weighted Averages](https://www.youtube.com/watch?v=lWzo8CajF5s)
- [Gradient Descent With Momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I)
- [RMSProp](https://www.youtube.com/watch?v=_e-LFe_igno)
- [Adam Optimization Algorithm](https://www.youtube.com/watch?v=JXQT_vxqwIs)
- [Learning Rate Decay](https://www.youtube.com/watch?v=QzulmoOg2JE)
- [Normalizing Activations in a Network](https://www.youtube.com/watch?v=tNIpEZLv_eg)
- [Fitting Batch Norm Into Neural Networks](https://www.youtube.com/watch?v=em6dfRxYkYU)
- [Why Does Batch Norm Work?](https://www.youtube.com/watch?v=nUUqwaxLnWs)
- [Batch Norm At Test Time](https://www.youtube.com/watch?v=5qefnAek8OA)
- [The Problem of Local Optima](https://www.youtube.com/watch?v=fODpu1-lNTw)

**API references:**

- [numpy.random.permutation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html) (R1)
- [tf.nn.moments](https://www.tensorflow.org/api_docs/python/tf/nn/moments) (R2)
- [tf.keras.optimizers.SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (R3)
- [tf.keras.optimizers.RMSprop](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop) (R4)
- [tf.keras.optimizers.Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) (R5)
- [tf.nn.batch_normalization (r2.6)](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/batch_normalization) (R6)
- [tf.keras.optimizers.schedules.InverseTimeDecay](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay) (R7)
- [deeplearning.ai](https://www.deeplearning.ai/)
