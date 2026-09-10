# Regularization

Preventing overfitting in neural networks with L2 weight decay, inverted dropout, early stopping, and data augmentation — from numpy backprop to Keras layers.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | What regularization is and why it reduces overfitting (penalty vs. algorithm change) |
| 2 | L2 (ridge/weight decay) cost, gradient term, and why biases are excluded |
| 3 | L1 vs. L2: sparsity vs. smooth shrinkage |
| 4 | Inverted dropout: masks, `keep_prob`, rescaling, and train-vs-test behavior |
| 5 | How dropout gates both forward and backward passes |
| 6 | Early stopping: validation monitoring with `threshold`, `patience`, `count` |
| 7 | Data augmentation as regularization-by-data |
| 8 | Keras mapping: `kernel_regularizer`, `model.losses`, `Dropout(rate, training)` |

---

## Task-by-Task Reference

---

### Task 0 — L2 cost, numpy (`0-l2_reg_cost.py`)

**Challenge:** Add the L2 penalty to a plain loss without touching biases.

**Approach:** Sum squared entries of every `W{l}` (Frobenius norm squared) and add `(lambtha/(2*m)) * sigma2` to `cost`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.sum(W**2)` over `W1..WL` | Frobenius-norm-squared penalty; biases (`b`) skipped |
| `lambtha/(2*m)` scaling | `1/m` averages over examples; `/2` cancels on differentiation |
| `lambtha` spelling | `lambda` is a Python keyword |

> **Key takeaway:** Monitored cost = data loss + weight-size penalty; big weights cost extra.

---

### Task 1 — L2 gradient descent, numpy (`1-l2_reg_gradient_descent.py`)

**Challenge:** Propagate the penalty through backprop — nothing else changes structurally.

**Approach:** Standard `dZ = AL - Y` loop from `L..1`; add `(lambtha/m)*W` to `dW` only, keep `tanh'` as `(1 - A_prev**2)`, propagate with the pre-update `W`, update in place.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `dW = dW_unreg + (lambtha/m)*W`, `db` unchanged | Derivative of the Task-0 penalty; biases not regularized |
| `W*(1 - alpha*lambtha/m)` reading of the update | Weight-decay intuition: shrink-then-step |
| `dA_prev = W.T @ dZ` before `weights[...] -= ...` | Must use original `W` for propagation |

> **Key takeaway:** L2 in the cost becomes a shrinkage term in the gradient.

---

### Task 2 — L2 cost, TensorFlow (`2-l2_reg_cost.py`)

**Challenge:** Same penalty, but Keras already collected it per layer.

**Approach:** One line: `return cost + model.losses`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.losses` | List of per-layer `kernel_regularizer` penalty tensors; adding a tensor broadcasts the sum |

> **Key takeaway:** In TF the penalty lives on the layers; total cost is data cost plus collected losses.

---

### Task 3 — L2 layer, TensorFlow (`3-l2_reg_create_layer.py`)

**Challenge:** Attach the penalty at layer-creation time.

**Approach:** `Dense(units=n, activation=activation, kernel_initializer=He, kernel_regularizer=L2(lambtha))(prev)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')` | He initialization for `tanh`/ReLU stacks |
| `tf.keras.regularizers.L2(lambtha)` as `kernel_regularizer` | Puts this layer's penalty into `model.losses`; kernel-only, bias excluded |

> **Key takeaway:** Regularize the kernel at construction; the cost function just sums what's collected.

---

### Task 4 — Dropout forward prop, numpy (`4-dropout_forward_prop.py`)

**Challenge:** Train a different thinned network each step while keeping test time deterministic.

**Approach:** Hidden layers: `tanh`, then Bernoulli mask `rand(A.shape) < keep_prob`, `A *= mask`, `A /= keep_prob` (inverted dropout), cache `D{l}`; output layer: `softmax`, no mask.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Per-neuron mask `~ Bernoulli(keep_prob)` | Random thinning; per-layer drop count `~ Binomial` |
| `/ keep_prob` rescaling | Preserves expected activation so test needs no scaling |
| `cache['D{l}']` for `1..L-1` only | Forward and backward share the mask |

> **Key takeaway:** Zero some units, boost survivors — ensemble effect without the ensemble cost.

---

### Task 5 — Dropout gradient descent, numpy (`5-dropout_gradient_descent.py`)

**Challenge:** Mirror the forward mask in the backward pass.

**Approach:** Same loop as Task 1 minus the L2 term; when `lyr > 1`, `dA_prev = W.T @ dZ`, then `dA_prev *= D{lyr-1}; dA_prev /= keep_prob`, then `dZ = dA_prev * (1 - A_prev**2)`; output layer unmasked; in-place updates.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `dA *= D / keep_prob` with `D = cache['D{lyr-1}']` | Dropped units get zero gradient; scaling matches forward |
| Mask index tied to the layer propagated *into* | Off-by-one (`D{lyr}` vs `D{lyr-1}`) is the classic bug |

> **Key takeaway:** Dead forward units must stay dead backward — gate `dA` with the same mask.

---

### Task 6 — Dropout layer, TensorFlow (`6-dropout_create_layer.py`)

**Challenge:** One helper that builds `Dense → Dropout` and respects inference mode.

**Approach:** He-initialized `Dense` (activation inside), then `Dropout(rate=1-keep_prob)(dense_out, training=training)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `rate = 1 - keep_prob` | Keras speaks `rate` = P(drop); intranet speaks `keep_prob` = P(keep) |
| `training=training` passthrough | Dropout active in training, deterministic passthrough at test |

> **Key takeaway:** Convert the convention (`rate` vs `keep_prob`) and always forward the `training` flag — `tf.nn.dropout` without it drops even at inference.

---

### Task 7 — Early stopping (`7-early_stopping.py`)

**Challenge:** Turn a noisy validation curve into a stop/continue decision.

**Approach:** `count = 0 if opt_cost - cost > threshold else count + 1`; `return (count >= patience, count)`. Pure Python, no numpy; caller owns `opt_cost` and best-weights restore.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Strict `opt_cost - cost > threshold` | Only a *real* drop resets patience; equality counts as flat |
| `(should_stop, count)` tuple | Stopping time as capacity knob; stopping weights are past-peak by definition |

> **Key takeaway:** Patience absorbs the bounce, threshold defines "real" — stop at the dev-error minimum, keep the best weights.

---

### Task 8 — Blog post (`blog.md`)

**Challenge:** Compare the toolbox in one page: L1/L2, dropout (+MC dropout/Bayesian note), augmentation, early stopping, plus the Adam caveat.

**Approach:** Mechanics → pros → cons per technique, closing with L2+SGD vs. AdamW guidance.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Augmentation (flip/rotate/scale/crop, offline vs. online) | Cheap data that teaches invariance |
| `AdamW` over `L2+Adam` | Decoupled weight decay; plain L2 interacts badly with adaptive rates |

> **Key takeaway:** One tool per job — penalties shrink weights, dropout breaks co-adaptation, augmentation manufactures data, stopping caps time.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `sum(W**2)`, `lambtha/(2m)` penalty, biases excluded | L2 cost (numpy) |
| 1 | `+(lambtha/m)W` in `dW`, `tanh'` backprop, in-place update | L2 gradient (numpy) |
| 2 | `cost + model.losses` | L2 cost (TF) |
| 3 | `Dense(..., kernel_regularizer=L2)`, He `VarianceScaling` | L2 layer (TF) |
| 4 | Bernoulli mask, inverted `/keep_prob`, `cache D`, softmax output untouched | Dropout forward (numpy) |
| 5 | `dA *= D/keep_prob` with `D{lyr-1}` | Dropout backward (numpy) |
| 6 | `Dropout(rate=1-keep_prob)(x, training=training)` | Dropout layer (TF) |
| 7 | `threshold/patience/count` strict-`>` rule → `(bool, count)` | Early stopping |
| 8 | Augmentation offline/online, MC dropout, AdamW note | Survey |

---

## Resources

* [RESOURCES.md](./RESOURCES.md) — ingested summaries of every Read-or-watch item (Wikipedia regularization + early stopping, Analytics Vidhya survey, McCaffrey L2+backprop, Medium L1/L2 intuitions, Galeone dropout analysis, Nanonets augmentation, deeplearning.ai C2W1 L04–L08 transcripts).
* [Hands-On ML, Ch. 4 — Ridge regression](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — L2 cost/gradient, bias exclusion (pp. 184, 188).
* [Hands-On ML, Ch. 11 — Dropout / max-norm](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — thinning, inverted scaling, MC dropout (pp. 422–425).
* [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) · [numpy.random.binomial](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html)
* [tf.keras.regularizers.L2](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L2) · [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) · [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
* [Dropout paper (Srivastava et al., JMLR)](http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) · [Early Stopping — But When? (Prechelt)](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)
