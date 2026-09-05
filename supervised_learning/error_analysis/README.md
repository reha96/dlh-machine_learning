# Error Analysis — Supervised Learning

Confusion-matrix toolkit: build the matrix from one-hot labels and predictions, derive per-class sensitivity, precision, specificity and F1, then apply bias/variance diagnosis to remediation scenarios (intranet project 2295).

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Build a confusion matrix from one-hot true labels and predicted labels |
| 2 | Compute per-class sensitivity (recall): the fraction of each true class recovered |
| 3 | Compute per-class precision: the fraction of each predicted class that is correct |
| 4 | Compute per-class specificity: the fraction of true negatives correctly rejected |
| 5 | Combine precision and recall into the per-class F1 score (harmonic mean) |
| 6 | Choose an error-handling strategy from diagnosed error sources (quiz) |
| 7 | Compare and contrast classification metrics and when each matters (quiz) |

---

## Task-by-Task Reference

### Task 0 — Create Confusion (`0-create_confusion.py`)

**Challenge:** Turn two one-hot matrices into a (classes, classes) count of true-vs-predicted pairs without Python loops.

**Approach:** Return `np.matmul(labels.T, logits)` on `(m, classes)` one-hot inputs; entry `(i, j)` counts examples truly class `i` predicted as class `j`. Rows are truth, columns are predictions. Verified on `MNIST.npz` (`labels` (50000, 10) float64, `logits` (50000, 10) int64 one-hot) reproducing `confusion.npz` (`confusion` (10, 10), total 50000).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `labels.T @ logits` one-hot trick | Outer-count of co-occurring true/predicted classes in one matmul |
| Row-truth / column-prediction convention | Reading rule for every later metric (rows → FN, columns → FP) |

> **Key takeaway:** A confusion matrix is a single matmul of one-hot matrices; its diagonal holds the per-class true positives.

---

### Task 1 — Sensitivity (`1-sensitivity.py`)

**Challenge:** Measure, per class, how much of the true population the classifier finds.

**Approach:** Take `TP = np.diag(confusion)`, row sums minus `TP` as `FN`, return `TP / (TP + FN)`. Row-wise because false negatives share the true label (same row, wrong column).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.diag(confusion)` | Extract per-class true positives from the diagonal |
| Row sums minus diagonal (`axis=1`) | Per-class false negatives; sensitivity = recall = `TP/(TP+FN)` |

> **Key takeaway:** Sensitivity (recall) reads along rows: of everything truly class `i`, how much did we catch.

---

### Task 2 — Precision (`2-precision.py`)

**Challenge:** Measure, per class, how trustworthy a positive prediction is — the column mirror of Task 1.

**Approach:** Same diagonal `TP`, but column sums (`axis=0`) minus `TP` as `FP`; return `TP / (TP + FP)`. Only the summation axis is new versus Task 1.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Column sums minus diagonal (`axis=0`) | Per-class false positives; precision = `TP/(TP+FP)` |
| Row/column duality | Rows answer "did we find it", columns answer "was the alarm right" |

> **Key takeaway:** Precision reads down columns: of everything predicted as class `i`, how much really is class `i`.

---

### Task 3 — Specificity (`3-specificity.py`)

**Challenge:** Measure per-class true-negative rate, which needs the one quantity no row or column sum gives directly: `TN`.

**Approach:** Reuse `TP` (diag), `FN` (row sums), `FP` (col sums) from Tasks 1–2, then derive `TN = total − TP − FP − FN` (one-vs-rest: everything that is neither the class nor predicted as it) and return `TN / (TN + FP)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `TN = total − TP − FP − FN` | Only new count; completes the four cells of each one-vs-rest table |
| Specificity `TN/(TN+FP)` | Fraction of true negatives correctly rejected per class |

> **Key takeaway:** Specificity is precision's counterpart on negatives; `TN` is whatever is left after removing `TP`, `FP`, `FN` from the total.

---

### Task 4 — F1 Score (`4-f1_score.py`)

**Challenge:** Combine precision and recall into one per-class number that punishes a model good at only one of them.

**Approach:** Recompute `TP/FN/FP` as in Tasks 1–3, then return `TP / (TP + (FN + FP)/2)` — the harmonic-mean form `2PR/(P+R)` simplified to counts. Imports the sibling solutions with `__import__('1-sensitivity')` / `__import__('2-precision')` (numeric filenames are not valid module identifiers).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Harmonic mean `2PR/(P+R)` ≡ `TP/(TP+(FN+FP)/2)` | Single score that drops steeply if either precision or recall is low |
| `__import__('1-sensitivity')` cross-file reuse | Import a module whose filename starts with a digit |

> **Key takeaway:** F1 is the harmonic (not arithmetic) mean of precision and recall, so one-sided models score badly.

---

### Task 5 — Dealing with Error (`5-error_handling`)

**Challenge:** Map each of four bias/variance regimes to the correct remediation — a written quiz, not code (extensionless answer file, no `.py`).

**Approach:** One line per scenario, CSV alphabetical for multi-select: High Bias + High Variance → `A,B,D`; High Bias + Low Variance → `A,B,D`; Low Bias + High Variance → `B,C,E`; Low Bias + Low Variance → `F`. Options: A train more, B different architecture, C more data, D deeper network, E regularization, F nothing. Content matches both reference forks byte-for-byte.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Bias → capacity/training (`A,B,D`) | Underfitting is fought with more training, new architectures, deeper nets |
| Variance → data/regularization (`C,E`) | Overfitting is fought with more data and regularization |
| Healthy model → `F` (nothing) | Low bias + low variance needs no intervention |

> **Key takeaway:** Diagnose first (bias or variance?), then prescribe: capacity cures bias, data and penalties cure variance.

---

### Task 6 — Compare and Contrast (`6-compare_and_contrast`)

**Challenge:** Given training and validation confusion matrices plus human-level error (~14%), name the single most important issue: A high bias, B high variance, C nothing.

**Approach:** Answer `A` — training error itself sits far above the human level, so the bottleneck is capacity/training (bias), not the train–validation gap. Single-letter file, matching both reference forks byte-for-byte.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Human level as bias floor | Avoidable bias = training error − human error; nothing beats this baseline |
| Gap reading | Variance = validation − training; a small gap with a bad floor still means bias |

> **Key takeaway:** Compare training error to human level first (bias?) and only then the train–validation gap (variance?) — order matters.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `labels.T @ logits` one-hot matmul; row-truth/col-pred convention | Metrics / Confusion matrix |
| 1 | `np.diag` for TP; row sums for FN; recall `TP/(TP+FN)` | Metrics / Sensitivity |
| 2 | Column sums for FP; precision `TP/(TP+FP)`; row/col duality | Metrics / Precision |
| 3 | `TN = total−TP−FP−FN`; specificity `TN/(TN+FP)` | Metrics / Specificity |
| 4 | Harmonic mean `2PR/(P+R)`; `__import__` of digit-leading modules | Metrics / F1 |
| 5 | Bias/variance regimes → remediation map (A–F scenarios) | Quiz / Error strategy |
| 6 | Human-level floor + train–val gap reading → bias verdict | Quiz / Bias–variance diagnosis |

---

## Resources

> No intranet Read-or-watch list is captured for this project yet (needs one authenticated session — pending). General references only; see the project spec for the official list.

- [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- [numpy.diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html)
- [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [sklearn.metrics.precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
- [Precision and recall — Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
