# Math — Multivariate Probability

A progressive implementation of multivariate statistics — from mean vectors and covariance matrices to correlation and the multivariate normal distribution, building toward the Gaussian foundations of machine learning with NumPy.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Compute the mean vector of a multivariate dataset with `np.mean` and axis control |
| 2 | Compute the covariance matrix manually using the sample covariance formula (no `np.cov`) |
| 3 | Understand Bessel's correction ($n-1$) for an unbiased sample covariance estimate |
| 4 | Derive the correlation matrix from covariance via diagonal scaling |
| 5 | Fit a multivariate normal distribution from data using maximum likelihood estimation |
| 6 | Compute the multivariate normal PDF using the full formula with determinant and inverse |
| 7 | Validate inputs with cascading exception checks in a specific priority order |

---

## Data Conventions

This module uses two matrix layouts — a key learning point about data conventions in ML:

| Layout | Shape | Meaning | Used In |
|--------|-------|---------|---------|
| Samples × Features | $(n, d)$ | $n$ data points, $d$ dimensions — each row is a sample | Task 0, Task 1 |
| Features × Samples | $(d, n)$ | $d$ dimensions, $n$ data points — each column is a sample | `MultiNormal` class |

The transpose flips between conventions. Pay attention to which axis you compute the mean along and how the covariance matrix multiplication is ordered.

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from earlier tasks are not repeated.

---

### Task 0 — Mean & Covariance (`0-mean_cov.py`)

**Challenge:** Compute both the mean vector and covariance matrix of a multivariate dataset $X$ — implementing the sample covariance formula from scratch without using `np.cov`.

**Approach:** Given $X$ of shape $(n, d)$, compute the mean with `np.mean(X, axis=0, keepdims=True)` to get shape $(1, d)$. Center the data: $X - \mu$. The sample covariance is $\frac{1}{n-1} (X - \mu)^\top (X - \mu)$ — the matrix multiplication of the centered data's transpose with itself, scaled by Bessel's correction. Inputs are validated in order: 2D ndarray check → $n \ge 2$ check.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.mean(X, axis=0, keepdims=True)` | Compute mean along samples axis, preserving 2D shape $(1, d)$ |
| Data centering: $X - \mu$ | Subtract the mean from every row via NumPy broadcasting |
| `(X - \mu)^\top @ (X - \mu) / (n - 1)$ | Sample covariance as sum of outer products divided by degrees of freedom |
| Bessel's correction ($n-1$ denominator) | Unbiased estimate — sample covariance, not population |
| `isinstance(X, np.ndarray) and X.ndim != 2` | Validate 2D array before any computation |
| `n < 2` → `ValueError` | Require multiple data points for meaningful covariance |

> **Key takeaway:** The covariance matrix $\Sigma$ captures how each pair of dimensions varies together — $\Sigma_{ij}$ is the covariance between dimension $i$ and dimension $j$. Bessel's correction ($n-1$) gives an unbiased estimate of the population covariance. Without `keepdims=True`, the mean collapses to 1D, breaking the broadcasting in the centering step.

---

### Task 1 — Correlation Matrix (`1-correlation.py`)

**Challenge:** Convert a covariance matrix into a correlation matrix — normalizing covariances to the $[-1, 1]$ range so relationships between variables measured in different units become directly comparable.

**Approach:** Extract the diagonal variances $\sigma_i^2$ with `np.diagonal()`. Compute scaling factors $\sigma_i^{-1/2}$ (element-wise power). Build a diagonal matrix $D$ from these factors with `np.diagflat()`. The correlation matrix is the sandwich product $D \cdot C \cdot D$, which scales each covariance $\sigma_{ij}$ by $\frac{1}{\sqrt{\sigma_{ii} \cdot \sigma_{jj}}}$.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.diagonal(C)` | Extract the variance vector (diagonal of covariance matrix) |
| `diag ** (-1/2)` | Element-wise power — compute $1/\sqrt{\sigma_i^2}$ for each variance |
| `np.diagflat(scaling_factors)` | Build a diagonal matrix from a 1D array of scaling factors |
| $D \cdot C \cdot D$ sandwich product | Normalize every $(i,j)$ entry: $\rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii} \cdot \sigma_{jj}}}$ |
| Correlation always in $[-1, 1]$ | Standardized covariance — scale-invariant and unit-free |

> **Key takeaway:** Correlation is standardized covariance — it strips away the units so you can compare relationships across variables measured on different scales (e.g., height in cm vs. weight in kg). The diagonal scaling trick $DCD$ is a clean one-liner that transforms any covariance matrix into its correlation form.

---

### Task (class) — Multivariate Normal Distribution (`multinormal.py`)

**Challenge:** Implement a class that fits a multivariate normal distribution from data and computes the PDF for any point — combining mean, covariance, determinant, and inverse into the full multivariate Gaussian formula. This is the culmination of all multivariate probability concepts.

**Approach:** The constructor accepts data of shape $(d, n)$ (features × samples — note the transposed convention from Task 0). It computes the mean along axis 1 (per-feature) with `keepdims=True` for shape $(d, 1)$. The covariance is $\frac{1}{n-1} (X - \mu)(X - \mu)^\top$ — note the reversed multiplication order vs. Task 0 because of the transposed layout. The `pdf(x)` method computes the full multivariate normal density:

$$f(x) = (2\pi)^{-d/2} \, |\Sigma|^{-1/2} \, \exp\!\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$$

using `np.linalg.det()` for the determinant, `np.linalg.inv()` for the precision matrix, and `.item()` to extract the scalar result from the 1×1 matrix.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Class-based distribution modeling | Encapsulate parameters (mean, cov) and operations (pdf) in a reusable object |
| $(d, n)$ data convention | Features as rows, samples as columns — common in ML literature |
| `np.mean(data, axis=1, keepdims=True)` | Compute per-feature means, yielding shape $(d, 1)$ |
| $(X - \mu) @ (X - \mu)^\top$ for covariance | Correct multiplication order for $(d, n)$ layout |
| `np.linalg.det(self.cov)` | Determinant $|\Sigma|$ — measures the "volume" of the distribution |
| `np.linalg.inv(self.cov)` | Precision matrix $\Sigma^{-1}$ — used in the Mahalanobis distance |
| $(x - \mu)^\top \Sigma^{-1} (x - \mu)$ | Mahalanobis distance — "how many std devs away" accounting for correlations |
| `.item()` on 1×1 array | Extract a Python float from a NumPy array of shape $(1, 1)$ |
| $(2\pi)^{-d/2}$ normalization factor | Ensures the PDF integrates to 1 over $\mathbb{R}^d$ |

> **Key takeaway:** The multivariate normal PDF generalizes the 1D Gaussian bell curve to $d$ dimensions. The covariance matrix $\Sigma$ controls the shape (spread) and orientation (correlation) of the bell. The Mahalanobis distance $(x - \mu)^\top \Sigma^{-1} (x - \mu)$ inside the exponent measures distance accounting for correlations — points on the same density contour have the same Mahalanobis distance. The determinant $|\Sigma|$ in the normalization scales the peak height: larger determinant → more spread → lower peak.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `np.mean` with axis/keepdims, data centering, Bessel's correction $(n-1)$, covariance via $(X-\mu)^\top(X-\mu)$ | Mean & Covariance |
| 1 | `np.diagonal()`, `np.diagflat()`, $DCD$ sandwich, correlation = covariance ÷ (std × std) | Correlation |
| class | Class-based distribution, $(d,n)$ layout, `np.linalg.det`/`inv`, Mahalanobis distance, multivariate PDF formula | Multivariate Normal |

---

## Multivariate Probability Pipeline

```
Data X ──→ Mean μ ──→ Covariance Σ ──→ Correlation P ──→ MultiNormal(μ, Σ) ──→ PDF(x)
  (n×d)    (1×d)        (d×d)            (d×d)              class               scalar
```

1. **Mean** locates the center of the data in $d$-dimensional space
2. **Covariance** captures the spread and pairwise relationships
3. **Correlation** standardizes to $[-1, 1]$ for interpretability
4. **MultiNormal** packages mean and covariance into a full probability distribution
5. **PDF** answers: "how likely is this point under the fitted distribution?"

---

## Resources

- [NumPy mean — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
- [NumPy linalg.det — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)
- [NumPy linalg.inv — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)
- [NumPy diagonal — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html)
- [Covariance Matrix — Wikipedia](https://en.wikipedia.org/wiki/Covariance_matrix)
- [Correlation — Wikipedia](https://en.wikipedia.org/wiki/Correlation)
- [Multivariate Normal Distribution — Wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
- [Mahalanobis Distance — Wikipedia](https://en.wikipedia.org/wiki/Mahalanobis_distance)