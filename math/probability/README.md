# Math — Probability Distributions

A progressive study of four fundamental probability distributions implemented as Python classes — binomial, normal (Gaussian), Poisson, and exponential — with parameter estimation from data and PMF/PDF/CDF computation.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Implement the binomial distribution: Bernoulli trials, $n$ and $p$ parameters |
| 2 | Implement the normal (Gaussian) distribution: mean $\mu$, standard deviation $\sigma$ |
| 3 | Implement the Poisson distribution: rate parameter $\lambda$, counting processes |
| 4 | Implement the exponential distribution: rate parameter $\lambda$, waiting times |
| 5 | Estimate distribution parameters from data using the method of moments |
| 6 | Compute PMF (probability mass function) for discrete distributions |
| 7 | Compute PDF (probability density function) for continuous distributions |
| 8 | Compute CDF (cumulative distribution function) for all four distributions |
| 9 | Convert between z-scores and x-values on the normal curve |

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from earlier tasks are not repeated.

---

### Task 0 — Binomial Distribution (`binomial.py`)

**Challenge:** Model the number of successes in $n$ independent Bernoulli trials, each with probability $p$ — implementing the binomial PMF from scratch using combinatorial formulas.

**Approach:** The constructor accepts either explicit $n$ and $p$ or estimates them from data. From data, compute the mean, then variance, then solve for $p = 1 - \sigma^2/\mu$ and $n = \text{round}(\mu/p)$. The PMF computes ${n \choose k} p^k (1-p)^{n-k}$ using iterative factorial accumulation to avoid overflow.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Method of moments estimation | Estimate $n$ and $p$ from sample mean and variance |
| Iterative binomial coefficient | Compute ${n \choose k}$ without factorials via product |
| `round()` vs `int()` for parameter estimation | Round $n$ to nearest integer (not truncate) |
| `self.p = float(p)`, `self.n = int(n)` | Explicit type casting for distribution parameters |

> **Key takeaway:** The binomial distribution models "number of successes in $n$ trials." Parameters can be estimated from data: $p = 1 - \text{variance}/\text{mean}$, then $n = \text{round}(\text{mean}/p)$.

---

### Task 1 — Normal Distribution (`normal.py`)

**Challenge:** Model the bell-shaped Gaussian distribution and compute probabilities on it — implementing the PDF formula $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ from scratch.

**Approach:** Store $\mu$ (mean) and $\sigma$ (stddev) as floats. Provide `z_score(x)` to convert x-values to z-scores, `x_value(z)` to convert back, `pdf(x)` for the density, and `cdf(x)` using the error function approximation. Class constants `e` and `pi` are hardcoded for precision control.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `z = (x - mean) / stddev` | Standardize a value to z-score (number of stddevs from mean) |
| `x = stddev * z + mean` | Reverse standardization: z-score back to raw value |
| Gaussian PDF formula | $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ |
| CDF via error function approximation | Compute cumulative probability using polynomial approximation of erf |
| Class constants `e` and `pi` | Pre-defined mathematical constants at module level |

> **Key takeaway:** The normal distribution is defined by $\mu$ (center) and $\sigma$ (spread). Z-scores standardize any normal to $\mathcal{N}(0,1)$. The CDF answers "what's the probability of being below x?"

---

### Task 2 — Poisson Distribution (`poisson.py`)

**Challenge:** Model the number of events occurring in a fixed interval — implementing the Poisson PMF $P(k) = \frac{\lambda^k e^{-\lambda}}{k!}$ and CDF as a sum.

**Approach:** The rate parameter $\lambda$ (lambtha) is either given or estimated as the sample mean. The PMF computes $\lambda^k e^{-\lambda} / k!$ using iterative factorial accumulation. The CDF sums PMF values from $0$ to $k$ using the same iterative factorial approach for efficiency.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| $\lambda = \frac{1}{n}\sum x_i$ | Estimate Poisson rate as the arithmetic mean of the data |
| Iterative $k!$ accumulation | Compute factorial incrementally to avoid recomputation |
| CDF = $\sum_{j=0}^{k} \text{PMF}(j)$ | Cumulative probability is the sum of individual PMF values |

> **Key takeaway:** The Poisson distribution models count data — "how many events in a fixed interval?" $\lambda$ is both the mean AND the variance. The PMF uses $e^{-\lambda}$ as the base probability of zero events.

---

### Task 3 — Exponential Distribution (`exponential.py`)

**Challenge:** Model the waiting time between events in a Poisson process — implementing the exponential PDF $f(x) = \lambda e^{-\lambda x}$ and CDF $F(x) = 1 - e^{-\lambda x}$.

**Approach:** The rate $\lambda$ is either given or estimated as $1/\text{mean}$ of the data (the reciprocal of the sample mean). The PDF computes $\lambda e^{-\lambda x}$ directly. The CDF uses $1 - e^{-\lambda x}$ — a simple closed form, unlike the Poisson which requires summation.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| $\lambda = 1 / \bar{x}$ | Estimate exponential rate as reciprocal of sample mean |
| Exponential PDF: $\lambda e^{-\lambda x}$ | Memoryless continuous distribution for waiting times |
| Exponential CDF: $1 - e^{-\lambda x}$ | Closed-form cumulative probability — no summation needed |

> **Key takeaway:** The exponential distribution is the continuous counterpart to the discrete Poisson. It models waiting times with the "memoryless" property: $P(X > s+t \mid X > s) = P(X > t)$. The rate $\lambda$ is the inverse of the expected waiting time.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | Binomial PMF, method of moments for $n$ and $p$ | Discrete Distributions |
| 1 | Gaussian PDF/CDF, z-score standardization, erf approximation | Continuous Distributions |
| 2 | Poisson PMF/CDF, $\lambda$ as rate, iterative factorial summation | Discrete Distributions |
| 3 | Exponential PDF/CDF, $\lambda = 1/\bar{x}$, memoryless property | Continuous Distributions |

---

## Resources

- [SciPy Stats — Probability Distributions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Binomial Distribution — Wikipedia](https://en.wikipedia.org/wiki/Binomial_distribution)
- [Normal Distribution — Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)
- [Poisson Distribution — Wikipedia](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Exponential Distribution — Wikipedia](https://en.wikipedia.org/wiki/Exponential_distribution)