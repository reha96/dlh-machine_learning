# Unsupervised Learning — Dimensionality Reduction

Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor
Embedding (t-SNE): two ways to compress high-dimensional data into fewer
dimensions, built from scratch in NumPy.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | What is dimensionality reduction and why it matters |
| 2 | Variance, covariance matrix, eigenvalues and eigenvectors |
| 3 | PCA via SVD: singular values, right singular vectors, the weights matrix W |
| 4 | Keeping variance: cumulative variance fractions and component counts |
| 5 | t-SNE's P affinities: Gaussian kernels, perplexity, Shannon entropy |
| 6 | Binary search: tuning each Gaussian's precision to a target entropy |
| 7 | Symmetrized affinities and the Q affinities with the Student-t kernel |
| 8 | Gradient descent with momentum and early exaggeration |
| 9 | KL divergence as the t-SNE cost function |
| 10 | The full t-SNE pipeline: PCA → P → optimize Y → cost reporting |

---

## Task-by-Task Reference

Each task entry captures only what is new relative to all previous tasks.

---

### Task 0 — PCA with variance preservation (`0-pca.py`)

**Challenge:** Find the weights matrix W whose columns pick the directions of
maximal variance, keeping only enough components to retain `var` (default
0.95) of the total variance.

**Approach:** Decompose X with `np.linalg.svd(X, full_matrices=False)`. The
squared singular values are proportional to the covariance eigenvalues, so
their cumulative sum over the total gives the variance fraction each prefix
of components keeps. `np.argmax(frac >= var) + 2` gives the component count —
one extra past the crossing, as the reference does — and `W = Vt[:nd].T`
extracts the dominant right singular vectors.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.linalg.svd(X, full_matrices=False)` | Factor X = U·diag(S)·Vt; Vt rows are the principal directions |
| `S * S` | Squared singular values ≈ covariance eigenvalues |
| `np.cumsum(S2) / S2.sum()` | Cumulative fraction of variance per component count |
| `np.argmax(frac >= var)` | First component that crosses the variance threshold |
| `Vt[:nd].T` | Build W (d, nd) from the nd dominant directions |

> **Key takeaway:** PCA is an eigendecomposition of the covariance in
> disguise — SVD gives the same directions without ever building the
> covariance matrix, and the singular values tell you how much variance each
> component carries.

---

### Task 1 — PCA with fixed dimensions (`1-pca.py`)

**Challenge:** Instead of a variance target, project onto a fixed number of
components `ndim`, and make the function validate its own input.

**Approach:** `ndim` must be a positive int (else `ValueError`). Center X
(`X - np.mean(X, axis=0)`) because this time the data is not pre-centered,
SVD again for the directions, clamp `ndim` to the components actually
available with `min(ndim, Vt.shape[0])`, then project:
`T = X_c @ Vt[:ndim].T`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `X - np.mean(X, axis=0)` | Center each feature in place |
| `raise ValueError(...)` | Reject invalid `ndim` instead of failing silently |
| `ndim = min(ndim, Vt.shape[0])` | Cap the request at available components |
| `X_c @ Vt[:ndim].T` | Project the centered data onto the chosen directions |

> **Key takeaway:** Projection is one matmul — the work is picking the right
> directions; the rest is guarding the input.

---

### Task 2 — Initialize P affinities (`2-P_init.py`)

**Challenge:** Prepare every variable t-SNE needs for its high-dimensional
similarities — distances, the empty affinity matrix, per-point Gaussian
precisions, and the entropy target — with no loops.

**Approach:** Compute pairwise squared distances with the identity
`‖x_i−x_j‖² = ‖x_i‖² + ‖x_j‖² − 2·x_i·x_j`, using `X @ X.T` for all dot
products and the row-norms `sum_X`. Zero the diagonal, then hand back an
`(n,n)` zeros matrix for P, `(n,1)` ones for betas, and the target entropy
`H = log2(perplexity)` (perplexity is defined as 2^H).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.sum(np.square(X), axis=1)` | Squared norm of each point |
| `-2 * np.dot(X, X.T)` | All pairwise dot products at once |
| `np.fill_diagonal(D, 0)` | Force self-distances to exactly 0 |
| `np.log2(perplexity)` | Target Shannon entropy every Gaussian must reach |

> **Key takeaway:** The whole pairwise distance matrix is three array
> expressions — no nested loops over pairs.

---

### Task 3 — Shannon entropy and P affinities (`3-entropy.py`)

**Challenge:** For one point, turn its distances into probabilities and
measure how "spread out" they are — the quantity the perplexity controls.

**Approach:** Apply the Gaussian kernel `num = exp(-Di * beta)`, normalize to
probabilities `Pi = num / num.sum()`, then compute the Shannon entropy
`Hi = -Σ Pi·log2(Pi)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.exp(-Di * beta)` | Gaussian weight for each distance; beta = precision |
| `num / np.sum(num)` | Normalize weights into a probability distribution |
| `-np.sum(Pi * np.log2(Pi))` | Shannon entropy in bits (log base 2) |

> **Key takeaway:** A bigger beta tightens the Gaussian, concentrates the
> probabilities, and lowers the entropy — that inverse relationship is what
> the next task exploits.

---

### Task 4 — P affinities (`4-P_affinities.py`)

**Challenge:** Give every point its own Gaussian so that all entropies equal
the target H — a root-finding problem solved per point, then fuse the
one-directional similarities into a symmetric affinity matrix.

**Approach:** For each point, drop its self-distance
(`np.delete(D[i], i)`), compute the entropy, and run a binary search on beta
until `|Hdiff| <= tol` (capped at 50 tries): entropy too high → double beta,
too low → halve it, otherwise bisect between the recorded bounds. Store the
row of Pi, then average the matrix with its transpose and divide by `2n`:
`P = (P + P.T) / (2 * n)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.delete(D[i], i)` | Distances from point i to all points but itself |
| Binary search over beta | Find the precision whose entropy matches H |
| `low` / `high` bounds + doubling/halving | Expand or bisect the search range |
| `P = (P + P.T) / (2 * n)` | Symmetrize p_j\|i into p_ij and normalize |

> **Key takeaway:** Perplexity is a constraint, not a parameter — each point's
> Gaussian is solved for, one binary search at a time.

---

### Task 5 — Q affinities (`5-Q_affinities.py`)

**Challenge:** Measure the same similarities in the low-dimensional space,
using a distribution with heavier tails than the Gaussian.

**Approach:** Same distance trick on Y, then the Student-t kernel
`num = 1 / (1 + D)` — the heavy tails leave more room between clusters in the
embedding. Zero the diagonal and normalize: `Q = num / num.sum()`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `1 / (1 + D)` | Student-t kernel with one degree of freedom |
| Return `(Q, num)` | Keep the unnormalized kernel for the gradient step later |

> **Key takeaway:** Q has the same pipeline as P — kernel, zero diagonal,
> normalize — only the kernel changes.

---

### Task 6 — Gradient descent (`6-grads.py`)

**Challenge:** Decide how each embedded point should move so Q comes closer
to P.

**Approach:** `PQ = P - Q` marks where the distributions disagree, and the
kernel weights `num` from Q_affinities scale each pair. For each point i,
`dY[i] = Σ_j PQ(j,i)·num(j,i)·(Y[i] − Y[j])`: attracted to neighbors P values
higher, repelled where Q dominates.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `PQ = P - Q` | Pairwise disagreement signal |
| `(PQ[:, i, np.newaxis] * num[:, i, np.newaxis]) * (Y[i] - Y)` | Weighted pull per pair |
| `.sum(axis=0)` | Accumulate the pull of every j onto point i |

> **Key takeaway:** The gradient is a sum of pairwise forces — each pair
> either pulls or pushes, scaled by how much the two distributions differ.

---

### Task 7 — Cost (`7-cost.py`)

**Challenge:** Score how well the embedding matches the original data, so the
training loop has a number to report.

**Approach:** Floor both matrices with `np.maximum(P, 1e-12)` (and Q) to keep
logs finite, then sum the KL divergence `C = Σ P·log(P/Q)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.maximum(P, 1e-12)` | Floor so `log(P/Q)` never sees 0 |
| `np.sum(P * np.log(P / Q))` | KL divergence of P against Q |

> **Key takeaway:** Cost is 0 when P equals Q and grows as they drift apart —
> it is the "distance" between the two distributions.

---

### Task 8 — t-SNE (`8-tsne.py`)

**Challenge:** Chain everything into one transform: PCA preprocessing,
affinity building, and a momentum-driven optimization loop with early
exaggeration and periodic cost reports.

**Approach:** PCA to `idims`, then `P × 4` so clusters separate hard in the
first 100 iterations (early exaggeration). Initialize Y randomly, keep the
previous step for momentum (0.5 for the first 20 iterations, 0.8 after), and
update `Y = Y − lr·dY + momentum·(Y − Y_prev)`, re-centering Y each step.
Stop exaggerating at iteration 100 and print the cost every 100 iterations.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `P = P * 4` / `P = P / 4` | Early exaggeration forces early separation |
| `Y - lr * dY + momentum * (Y - Y_prev)` | Gradient step with a momentum term |
| `if i < 20: momentum = 0.5 else 0.8` | Low momentum first, higher after warm-up |
| `Y - np.mean(Y, axis=0)` | Keep the embedding centered on the origin |

> **Key takeaway:** The algorithm is a short loop — gradients, momentum,
> center, report — and every piece was already built in tasks 0–7.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | SVD, squared singular values, cumulative variance, `W = Vt[:nd].T` | PCA / SVD |
| 1 | Centering, `ndim` clamp, `X_c @ Vt[:ndim].T` projection | PCA / SVD |
| 2 | Pairwise distance trick, `log2(perplexity)` target, init arrays | Model init |
| 3 | Gaussian kernel, normalize, Shannon entropy | Entropy & kernels |
| 4 | Per-point binary search on beta, symmetrize `(P+P.T)/2n` | Optimization |
| 5 | Student-t kernel `1/(1+D)`, `(Q, num)` pair | Entropy & kernels |
| 6 | `dY[i] = Σ_j (P−Q)·num·(Y_i−Y_j)` pairwise forces | Gradient descent |
| 7 | `1e-12` floor, KL divergence `Σ P·log(P/Q)` | Evaluation |
| 8 | Early exaggeration, momentum switch, centering, cost loop | Pipeline |

---

## Resources

- [t-SNE paper — van der Maaten & Hinton](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- [PCA — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [t-SNE — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [np.linalg.svd — NumPy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
- [Shannon entropy — Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Kullback-Leibler divergence — Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- [Principal component analysis — Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Student's t-distribution — Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
