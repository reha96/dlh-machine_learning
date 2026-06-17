# Math — Bayesian Probability

A progressive implementation of Bayes' theorem — building the complete pipeline from likelihood through posterior probability, using a drug trial scenario with NumPy vectorization.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Compute the binomial likelihood $P(\text{data} \mid P)$ for a vector of hypothetical probabilities |
| 2 | Calculate the intersection $P(\text{data} \mid P) \cdot P(P)$ — likelihood weighted by prior belief |
| 3 | Compute the marginal probability $P(\text{data})$ as the sum of all intersections |
| 4 | Derive the posterior $P(P \mid \text{data})$ via Bayes' theorem: intersection ÷ marginal |
| 5 | Understand the Bayesian workflow: prior → likelihood → marginal → posterior |
| 6 | Apply NumPy vectorized operations on 1D arrays for efficient probability calculations |
| 7 | Validate inputs with cascading exception checks in a specific order |

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from earlier tasks are not repeated.

---

### Task 0 — Likelihood (`0-likelihood.py`)

**Challenge:** Compute the binomial likelihood $P(x \mid n, P)$ for an entire vector of hypothetical probabilities — the first step in any Bayesian analysis.

**Approach:** For a drug trial with $n$ patients and $x$ side-effect cases, compute ${n \choose x} P^x (1-P)^{n-x}$ for every probability in the NumPy array $P$. The binomial coefficient is computed iteratively to avoid factorials. All operations are vectorized over $P$.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Binomial likelihood formula ${n \choose x} P^x (1-P)^{n-x}$ | Model the probability of observing $x$ events in $n$ trials |
| Iterative binomial coefficient computation | Avoid factorial overflow for large $n$ |
| NumPy array broadcasting `P ** x` | Apply exponentiation to every element simultaneously |
| Cascading validation with specific error messages | Validate inputs in a defined priority order |

> **Key takeaway:** The likelihood answers "how probable is this data under each hypothetical parameter value?" It's the engine of Bayesian inference — all downstream calculations depend on it.

---

### Task 1 — Intersection (`1-intersection.py`)

**Challenge:** Combine the likelihood with prior beliefs — computing $P(\text{data} \mid P) \cdot P(P)$, the joint probability of data and parameters.

**Approach:** Call `likelihood()` from Task 0, then multiply the result element-wise by the prior array `Pr`. Validate that `Pr` has the same shape as `P` and sums to 1 (using `np.isclose` for floating-point safety). The intersection represents the unnormalized posterior.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.shape(Pr) == np.shape(P)` | Validate that prior and hypothesis arrays align |
| `np.isclose(np.sum(Pr), 1.0)` | Check that prior probabilities sum to 1 (floating-point safe) |
| Element-wise array multiplication `Pr * likelihood` | NumPy vectorized product of two 1D arrays |
| Cross-file function dependency | `intersection()` depends on `likelihood()` from Task 0 |

> **Key takeaway:** The intersection is the numerator of Bayes' theorem: prior × likelihood. It's the "unnormalized posterior" — proportional to the posterior but not yet summing to 1.

---

### Task 2 — Marginal Probability (`2-marginal.py`)

**Challenge:** Compute the marginal probability $P(\text{data})$ — the normalizing constant that makes the posterior a valid probability distribution.

**Approach:** Call `intersection()` from Task 1, then sum all intersection values with `np.sum()`. The marginal is a single number — the total probability of observing the data under all hypotheses, weighted by prior beliefs. Bundles local copies of `likelihood()` and `intersection()` for self-containment.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.sum(intersection)` | Sum all intersection values to get the marginal probability |
| Marginal = $\sum P(\text{data} \mid P) \cdot P(P)$ | The law of total probability — integrate over all hypotheses |
| Self-contained module with bundled dependencies | Copy dependent functions to make the file standalone |

> **Key takeaway:** The marginal probability $P(\text{data})$ is the denominator of Bayes' theorem. It's a single number that normalizes the posterior: every possible way the data could occur, weighted by prior belief.

---

### Task 3 — Posterior Probability (`3-posterior.py`)

**Challenge:** Compute the posterior probability $P(P \mid \text{data})$ — the updated belief about each hypothesis after observing the data. This is the final output of Bayesian inference.

**Approach:** Call `intersection()` and `marginal()` from previous tasks, then divide: `posterior = intersection / marginal`. The result sums to 1 — it's a proper probability distribution. The posterior answers: "given the data, what should I now believe about each possible probability $P$?"

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `intersection / marginal` | Bayes' theorem: posterior = (prior × likelihood) ÷ marginal |
| Full Bayesian pipeline: prior → likelihood → posterior | End-to-end probabilistic inference in 4 composable steps |
| Posterior sums to 1 (verifiable with `np.sum()`) | Valid probability distribution — can be used for decision-making |

> **Key takeaway:** The posterior is the goal of Bayesian analysis — it tells you what to believe after seeing the data. The entire pipeline (likelihood → intersection → marginal → posterior) implements Bayes' theorem: $P(H \mid D) = \frac{P(D \mid H) \cdot P(H)}{P(D)}$.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | Binomial likelihood, iterative binomial coefficient, NumPy broadcasting | Likelihood |
| 1 | Element-wise prior × likelihood, `np.isclose` for sum-to-1 check | Intersection |
| 2 | `np.sum()` for marginal probability, law of total probability | Marginal |
| 3 | Bayes' theorem: posterior = intersection ÷ marginal, full pipeline | Posterior |

---

## Bayesian Pipeline (Conceptual)

```
Prior P(H)  ──┐
               ├──→ Intersection ──→ Marginal P(D) ──→ Posterior P(H|D)
Likelihood    ──┘     P(D|H)·P(H)      Σ P(D|H)·P(H)      Intersection/Marginal
P(D|H)
```

---

## Resources

- [Bayes' Theorem — Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [NumPy Array Operations](https://numpy.org/doc/stable/reference/arrays.html)
- [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
- [Think Bayes — Allen Downey](https://greenteapress.com/wp/think-bayes/)