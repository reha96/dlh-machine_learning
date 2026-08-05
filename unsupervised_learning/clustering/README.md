# Unsupervised Learning — Clustering

Clustering in NumPy from scratch: K-means, Gaussian Mixture Models, the
Expectation-Maximization algorithm, BIC model selection, and the sklearn/scipy
APIs that replace them.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | What is unsupervised learning and what is clustering |
| 2 | K-means: centroid initialization, assignment, update, empty-cluster handling |
| 3 | Choosing the number of clusters: variance analysis and the elbow method |
| 4 | The multivariate Gaussian distribution (mean vector, covariance, pdf) |
| 5 | Gaussian Mixture Model parameters: priors, means, covariances |
| 6 | EM expectation step: soft assignments via Bayes' theorem |
| 7 | EM maximization step: soft counts and weighted statistics |
| 8 | Log-likelihood convergence and early stopping |
| 9 | Bayesian Information Criterion for model selection |
| 10 | sklearn clustering APIs: KMeans, GaussianMixture |
| 11 | Agglomerative hierarchical clustering with scipy |

---

## Task-by-Task Reference

Each task entry captures only what is new relative to all previous tasks.

---

### Task 0 — Initialize K-means (`0-initialize.py`)

**Challenge:** Initialize `(k, d)` centroids without loops, with exactly one
`numpy.random.uniform` call, each dimension bounded by the data's range.

**Approach:** Take `X.min(axis=0)` and `X.max(axis=0)` (per-dimension ranges,
shape `(d,)`), then let NumPy broadcast them into a single
`np.random.uniform(mins, maxs, size=(k, d))` draw.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `X.min(axis=0)` / `X.max(axis=0)` | Per-dimension data bounds in one vectorized call |
| `np.random.uniform(mins, maxs, size=(k, d))` | Draw a full grid of centroids with per-dimension ranges, no loop |

> **Key takeaway:** Broadcasting turns a k-loop of per-dimension random draws
> into a single array expression.

---

### Task 1 — K-means (`1-kmeans.py`)

**Challenge:** Assign every point to its nearest centroid without per-point
loops; re-seed empty clusters; stop when centroids stop moving.

**Approach:** Compute all pairwise distances with broadcasting
(`X[:, None, :] - C`), take `argmin(axis=1)` for assignments, and recompute each
centroid as the boolean-mask mean of its points. A cluster with no points gets
re-seeded with `np.random.uniform(mins, maxs)`. Loop breaks when
`np.all(C == C_old)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `X[:, np.newaxis, :] - C` | Pairwise distance matrix via broadcasting: `(n, k, d)` |
| `.sum(axis=2)`, `.argmin(axis=1)` | Squared distances, then nearest-centroid index per point |
| `mask = (clss == j)`; `X[mask].mean(axis=0)` | Boolean-mask filtering to recompute a centroid |

> **Key takeaway:** The assignment step is a vectorized nearest-neighbour
> query; the loop budget goes to iterations and clusters, never to points.

---

### Task 2 — Variance (`2-variance.py`)

**Challenge:** Total intra-cluster variance with no loops.

**Approach:** Chain reductions in one expression:
`(diffs ** 2).sum(axis=2).min(axis=1).sum()` — squared distance to every
centroid, keep the smallest per point, sum over points.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `.min(axis=1)` after `.sum(axis=2)` | Per-point nearest-centroid squared distance |
| `var = ... .min(1).sum()` | Total intra-cluster variance in one chained expression |

> **Key takeaway:** Variance measures how tightly points hug their assigned
> centroid — the score that K-means minimizes.

---

### Task 3 — Optimize K (`3-optimum.py`)

**Challenge:** Find the "right" k by measuring how variance drops as clusters
grow; `kmax=None` defaults to the number of points; two loops max.

**Approach:** Sweep `k` from `kmin` to `kmax`, run K-means and variance for
each, then express every variance as the drop from the `kmin` baseline
(`d_vars`). The elbow — where extra clusters stop buying much variance — is
the answer.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Sweep loop over `range(kmin, kmax + 1)` | Run K-means once per candidate k |
| `d_vars.append(var_baseline - j)` | Delta-variance curve for elbow detection |
| `kmax = X.shape[0] if kmax is None` | Optional-argument default pattern |

> **Key takeaway:** More clusters always reduce variance; the elbow is where
> the reduction stops being worth it.

---

### Task 4 — Initialize GMM (`4-initialize.py`)

**Challenge:** Seed a Gaussian Mixture Model — priors, means, covariances —
with no loops, reusing K-means for the means.

**Approach:** `pi = np.full(k, 1/k)` (neutral priors), `m` from
`kmeans(X, k)[0]`, and covariances as identity matrices stacked to `(k, d, d)`
via `np.identity(d)[np.newaxis].repeat(k, axis=0)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.full(k, 1/k)` | Equal priors in one call |
| `np.newaxis` + `.repeat(k, axis=0)` | Stack a `(d, d)` matrix into `(k, d, d)` |

> **Key takeaway:** GMM initialization is K-means for the means plus neutral
> priors and identity covariances — EM does the rest.

---

### Task 5 — Multivariate Gaussian PDF (`5-pdf.py`)

**Challenge:** Vectorize the quadratic form `(x−μ)ᵀΣ⁻¹(x−μ)` for all points and
avoid numeric underflow in the exponent.

**Approach:** Compute the normalization constant from
`(2π)^(−d/2) · det(S)^(−1/2)`, evaluate the quadratic form with the
matmul-multiply trick `((X − m) @ np.linalg.inv(S)) * (X − m)` summed over
`axis=1`, and floor the result with `np.maximum(P, 1e-300)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.linalg.det(S)`, `np.linalg.inv(S)` | Covariance determinant and inverse |
| `(X−m) @ inv(S) * (X−m)` then `.sum(axis=1)` | Broadcast quadratic form over all points |
| `np.maximum(P, 1e-300)` | Underflow floor so downstream `log` never sees 0 |

> **Key takeaway:** The quadratic form is a row-wise dot product — express it
> as a matmul followed by an element-wise multiply to avoid a loop over points.

---

### Task 6 — Expectation step (`6-expectation.py`)

**Challenge:** Compute soft assignments `g[k, n] = π_k · pdf_k / Σ_j π_j ·
pdf_j` and the model log-likelihood, with one loop.

**Approach:** For each component, `g[i] = pi[i] * pdf(X, m[i], S[i])`; accumulate
the mixture evidence in `denominator`; normalize; then
`ll = np.sum(np.log(denominator))`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `g = g / denominator` | Bayes' rule: posterior = weighted likelihood / evidence |
| `np.sum(np.log(denominator))` | Log-likelihood of the model — logs of the *evidence*, not of g |
| `np.isclose(pi.sum(), 1.0)` | Float-safe validation of the prior constraint |

> **Key takeaway:** `ll` scores how well the mixture explains the data
> (`Σ log P(x)`); normalized `g` would give `log 1 = 0`, which is why the log
> is taken on the denominator.

---

### Task 7 — Maximization step (`7-maximization.py`)

**Challenge:** Update `(π, μ, Σ)` from the soft assignments — the M-step of EM.

**Approach:** For each component: `nk = g[i].sum()` (soft count `N_k`),
`pi[i] = nk / n`, `m[i] = g[i] @ X / nk`, and the weighted covariance
`(g[i][:, None] * diff).T @ diff / nk` with `diff = X − m[i]`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `nk = g[i].sum()` | Soft count of points in cluster i |
| `g[i] @ X / nk` | Weighted mean |
| `(g[i][:, None] * diff).T @ diff / nk` | Weighted covariance from weighted outer products |

> **Key takeaway:** Every M-step formula is the familiar statistic with hard
> assignments replaced by soft responsibilities.

---

### Task 8 — Expectation Maximization (`8-EM.py`)

**Challenge:** Orchestrate E and M steps until the log-likelihood stops moving
(`|Δll| ≤ tol`) or the budget runs out, with a verbose reporting protocol.

**Approach:** Initial E step, then a single loop: print every 10 iterations,
run M, run E, check `abs(ll − prev_ll) <= tol` and break. The final reported
`g`/`ll` come from the E step run *after* the last M step, so the return is
self-consistent.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `abs(ll - prev_ll) <= tol` | Early stopping on the likelihood change |
| E step → check → M step ordering | Break before wasting an M step on a converged model |
| Final print after the loop | "Log Likelihood after {n} iterations" protocol |

> **Key takeaway:** EM climbs the likelihood hill; when the altimeter stops
> moving by more than `tol`, you have arrived.

---

### Task 9 — Bayesian Information Criterion (`9-BIC.py`)

**Challenge:** Pick the best k by balancing fit against complexity: more
clusters always raise `ll`, so charge a parameter tax.

**Approach:** Sweep k, run EM for each, count free parameters
`p = (k−1) + k·d + k·d(d+1)/2` (priors minus one, means, symmetric
covariances), score `b = p·ln(n) − 2·ll`, and take `argmin`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `p = (k - 1) + k * d + k * d * (d + 1) // 2` | Free-parameter count for a GMM |
| `b = p * np.log(n) - 2 * ll` | BIC: fit reward minus complexity tax |
| `kmin + np.argmin(b)` | Position in the b array vs. the k value it stands for |

> **Key takeaway:** The tax `p·ln(n)` grows with k, so the best k is where the
> fit gain stops covering the complexity cost.

---

### Task 10 — K-means, sklearn (`10-kmeans.py`)

**Challenge:** Replace the hand-rolled K-means with scikit-learn's estimator.

**Approach:** `sklearn.cluster.KMeans(k).fit(X)`, then read `cluster_centers_`
and `labels_` — the same `C` and `clss` as before, via fit-then-read.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `KMeans(k).fit(X)` | Estimator fit in one chained call |
| `.cluster_centers_`, `.labels_` | Post-fit attribute access |

> **Key takeaway:** sklearn exposes results as fitted attributes, not return
> values — fit first, read afterwards.

---

### Task 11 — GMM, sklearn (`11-gmm.py`)

**Challenge:** Same pipeline for a Gaussian Mixture Model, plus a BIC score
straight from the library.

**Approach:** `sklearn.mixture.GaussianMixture(k).fit(X)`, read `weights_`,
`means_`, `covariances_`, assign with `predict(X)`, score with `bic(X)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `GaussianMixture(k).fit(X)` | Full GMM estimation with EM inside sklearn |
| `gm.predict(X)` | Hard assignments from the fitted posteriors |
| `gm.bic(X)` | BIC score — note it takes the data as an argument |

> **Key takeaway:** `bic` is a bound method: without `(X)` you get the
> function, not the score.

---

### Task 12 — Agglomerative (`12-agglomerative.py`)

**Challenge:** Hierarchical clustering with imports restricted to
`scipy.cluster.hierarchy` and `matplotlib.pyplot` — no sklearn, no own
algorithm.

**Approach:** Build the tree with `linkage(X, method='ward')`, cut it at the
cophenetic threshold with `fcluster(Z, dist, criterion='distance')`, and draw
it with `dendrogram(Z, color_threshold=dist)`. Cluster indices are 1-based by
scipy convention.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `scipy.cluster.hierarchy.linkage(X, 'ward')` | Build the merge tree (Ward's method) |
| `fcluster(Z, dist, 'distance')` | Cut the tree at a cophenetic distance |
| `dendrogram(Z, color_threshold=dist)` | Draw the tree, one color per cluster below the cut |

> **Key takeaway:** The whole hierarchy is one `linkage` call; `fcluster` is
> simply "cut the tree at this height".

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `np.random.uniform` with per-dimension bounds | Random init |
| 1 | Broadcasting pairwise distances, boolean-mask means | K-means |
| 2 | Chained `sum → min → sum` variance | K-means |
| 3 | k-sweep + delta-variance elbow | Model selection |
| 4 | `np.full` priors, `np.identity` + `repeat` covariances | GMM init |
| 5 | Vectorized quadratic form, `np.maximum` underflow floor | Multivariate Gaussian |
| 6 | Bayes soft assignments, log-evidence likelihood | GMM/EM |
| 7 | Soft counts, weighted mean and covariance | GMM/EM |
| 8 | `abs(Δll) <= tol` early stopping, verbose protocol | GMM/EM |
| 9 | Parameter count + `p·ln(n) − 2·ll` BIC | Model selection |
| 10 | `KMeans(k).fit()` → `cluster_centers_`/`labels_` | sklearn API |
| 11 | `GaussianMixture(k)` → `predict(X)`/`bic(X)` | sklearn API |
| 12 | `linkage` → `fcluster` → `dendrogram` | scipy hierarchy |

---

## Resources

- [scipy.cluster.hierarchy — Hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [NumPy Array Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Bayesian Information Criterion (Wikipedia)](https://en.wikipedia.org/wiki/Bayesian_information_criterion)
- [Multivariate normal distribution (Wikipedia)](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
