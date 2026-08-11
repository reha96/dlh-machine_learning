# Clustering — Resources

## Read or watch

## K-means clustering: how it works (Victor Lavrenko)

URL: https://www.youtube.com/watch?v=_aWzGGNrcic  ·  Date: 2014-01-19 (YouTube upload)  ·  Status: summary

K-means takes the data points plus a user-chosen K. It drops K centroids at random locations, then repeats two steps: assign every point to its nearest centroid (Euclidean or any distance that suits the data), then move each centroid to the mean of the points assigned to it. The loop stops when no point changes cluster. Because the centroid is an average, the attributes must be numeric — averaging "zoo" and "car" makes no sense. Complexity runs iterations × K × n × d, and the distance computation dominates; even so, k-means is the fastest clustering algorithm around.

- Alternating steps: nearest-centroid assignment, then centroid = mean of its cluster's points.
- Converges when assignments stop changing; every centroid shifts at every update, not just one.
- Requires numeric attributes only (categorical values cannot be averaged).
- Cost ≈ iterations × K clusters × n instances × d dimensions; the per-point distance pass dominates.

## How many clusters? (Victor Lavrenko)

URL: https://www.youtube.com/watch?v=xNfOheh-res  ·  Date: 2014-01-19 (YouTube upload)  ·  Status: summary

Choosing K is a genuinely open problem. If the data has a natural class count, use it (10 digits → K = 10). Otherwise variance keeps falling as K grows — it bottoms out at zero when K = n, and a validation set does not fix it, because extra centroids always saturate the space. The book's minimum-description-length criterion returns a number that is neither 1 nor n, but it is arbitrary and rarely a good practical clustering. The standard workaround is the scree plot: plot variance against K and pick the elbow where steep drops turn into marginal ones, which is the same as maximizing the second derivative of the curve. Every option is hacky; none is better.

- Natural K: use the known class count when one exists (e.g. 10 for handwritten digits).
- Variance → 0 as K → n; validation sets do not rescue K here the way they do in k-NN.
- MDL trades fit against bits to encode centroids, but the number it yields is arbitrary.
- Scree plot: the "mountain" of big drops meets the "rubble" of small ones at the elbow; the elbow maximizes the second derivative.

## Bimodal distribution (Victor Lavrenko)

URL: https://web.archive.org/web/20221008001918/https://www.youtube.com/watch?v=BWItfiVnDfU  ·  Date: 2014-01-19 (YouTube upload)  ·  Status: failed-transcript (video removed; archived copy only)

The original upload is unavailable, so only the archived page remains; the topic — a distribution with two modes, typically the mixture of two Gaussian bumps — is covered in the Multimodal distribution entry below.

## EM algorithm: how it works (Victor Lavrenko)

URL: https://www.youtube.com/watch?v=REypj2sy_5U  ·  Date: 2014-01-19 (YouTube upload)  ·  Status: summary

Mixture models are the probabilistically sound way to do soft clustering: every point belongs to every cluster, but with a different degree of belief. Each cluster is a generative distribution (a Gaussian for real-valued data, a multinomial for discrete), and the job is to infer its parameters. If you knew which points came from which Gaussian, the means and variances would be trivial sample statistics; if you knew the means and variances, Bayes' rule gives each point's membership. EM breaks this chicken-and-egg loop: place the Gaussians randomly, compute soft assignments from the current parameters, re-estimate the parameters from those assignments, and iterate.

- Mixture model = soft clustering; memberships stay probabilities, never quantized to 0/1.
- Each cluster is a full distribution; for Gaussians you fit means and covariances.
- Parameters imply memberships (Bayes); memberships imply parameters (averaging).
- EM alternates those two, like k-means, but with probabilities instead of hard assignments.

## Expectation Maximization: how it works (Victor Lavrenko)

URL: https://www.youtube.com/watch?v=iQoXFmbXRJA  ·  Date: 2014-01-19 (YouTube upload)  ·  Status: summary

A 1-D walk-through with two Gaussians. Start them at random positions and, for each point, compute the posterior b_i that it came from the blue Gaussian via Bayes; a_i = 1 − b_i is the yellow weight. These weights are a fractional coloring — k-means would paint a point fully blue or fully yellow, EM splits its mass. Re-estimate means and variances as weighted averages using those probabilities: a point near a Gaussian contributes almost its full weight there and a sliver elsewhere. Priors can also be estimated (mean of b_i), but one component often grabs everything early, so pinning priors to uniform is common.

- b_i = posterior for blue, a_i = 1 − b_i for yellow; soft fractional coloring of points.
- New mean = Σ b_i x_i / Σ b_i; if b_i ∈ {0,1} this collapses to the k-means update.
- Variances update the same way, weighting squared deviations from the new mean.
- Prior estimate = average of b_i; uniform priors are a common fix when a component swallows the data.

## Mixture Models 4: multivariate Gaussians (Victor Lavrenko)

URL: https://www.youtube.com/watch?v=zL_MHtT56S0  ·  Date: 2014-01-19 (YouTube upload)  ·  Status: summary

The 1-D recipe carries over to vectors: the mean becomes a responsibility-weighted average of the instance vectors, taken attribute by attribute. When the dimensionality is small you can also estimate a covariance matrix, which lets Gaussians tilt off the coordinate axes (oblique shapes). Each covariance entry sums, over points weighted by their responsibility for that component, the product of the point's deviation in attribute j and its deviation in attribute k — positive values mean the attributes rise together. The multivariate density puts the inverse covariance matrix between the vector deviations: (x − μ)ᵀ Σ⁻¹ (x − μ).

- Mean = weighted average of instance vectors, per attribute.
- Covariance matrix encodes attribute correlations and allows tilted, non-axis-aligned Gaussians.
- Σ entry ∝ Σᵢ γᵢ (x_ij − μ_j)(x_ik − μ_k).
- Multivariate density: inverse covariance sandwiched between the deviation vectors.

## Mixture Models 5: how many Gaussians? (Victor Lavrenko)

URL: https://www.youtube.com/watch?v=BWXd5dOkuTo  ·  Date: 2014-01-19 (YouTube upload)  ·  Status: summary

Picking K for a mixture is no easier than for k-means, but the probabilistic objective gives sharper tools: maximize the log-likelihood of the data under a K-component mixture. That peaks when every data point gets its own dedicated Gaussian, and train/validation splits still favor very large K in many domains. The book's answer is Occam's razor encoded as AIC/BIC: reward fit (likelihood L) and penalize model complexity (parameter count p — note the covariance matrix makes p quadratic in the dimension). BIC's number sounds principled but is still arbitrary; if the mixture feeds a classifier, pick K by that classifier's validation error instead.

- Likelihood keeps rising with K; the maximum sits at one Gaussian per data point.
- AIC/BIC = fit (L) vs complexity (p): means, variances, priors, and the quadratic covariance entries.
- Occam's razor: equal fit favors the simpler model.
- Practical rule: choose K via validation error of the downstream task, not BIC alone.

## Hierarchical Clustering (Artificial Intelligence - All in One)

URL: https://www.youtube.com/watch?v=rg2cjfMsCk4  ·  Date: 2016-04-13 (YouTube upload)  ·  Status: summary

Agglomerative (bottom-up) clustering starts with every point as its own cluster and repeatedly merges the two nearest clusters, recording each merge in a dendrogram; divisive clustering splits top-down instead. Three design questions drive it: how to represent a cluster (the centroid, an average, works in Euclidean space; non-Euclidean spaces need a clustroid — an actual existing point chosen by smallest max/mean/sum-of-squares distance to the others), how to measure cluster nearness, and when to stop (at a preset K, or before a merge would create a "bad" cluster judged by diameter, radius, or density thresholds). The naive algorithm costs O(n³); priority queues bring it to O(n² log n), still too heavy for datasets that do not fit in memory.

- Agglomerative merges the two closest clusters; the dendrogram shows the whole merge history.
- Centroids only exist in Euclidean space; elsewhere use a clustroid (a real point).
- Stop at fixed K or when cohesion (diameter / radius / density) would cross a threshold.
- Complexity: O(n³) naive, O(n² log n) with priority queues; meant for small in-memory data.

## Understanding K-means Clustering in Machine Learning (Education Ecosystem, TDS)

URL: https://medium.com/data-science/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1  ·  Date: 2018-09-12 (publication date)  ·  Status: summary

A beginner's tutorial. K-means is an unsupervised algorithm: you pick a target number of centroids k, assign each point to its nearest cluster center, and iterate until the centroids stabilize or the iteration cap is reached; the "means" are the averaging of the data that produces each centroid. The post then works through sklearn on two random 2-D blobs — fit, read cluster_centers_ and labels_, predict a test point — and closes with the usual caveats: it is fast and simple, but sensitive to data variation and assumes spherical, evenly sized clusters.

- "Means" = averaging data points to locate centroids; objective = reduce in-cluster sum of squares.
- Halts when centroids stabilize or the defined iteration count is reached.
- sklearn flow: KMeans(n_clusters=k).fit(X) → cluster_centers_, labels_, predict().
- Weaknesses: high variance across datasets; assumes spherical, evenly sized clusters.

## Gaussian Mixture Model (Brilliant — John McGonagle, Geoff Pilling, Andrei Dobre, et al.)

URL: https://brilliant.org/wiki/gaussian-mixture-model/  ·  Date: unknown  ·  Status: summary

A GMM models a population as a weighted sum of normally distributed subpopulations — heights, say, as a male Gaussian plus a female Gaussian, learned without gender labels. The univariate model is p(x) = Σ φ_k N(x|μ_k, σ_k) with weights summing to 1; the multivariate version uses means and covariance matrices. Maximum likelihood has no closed form for mixtures, so EM alternates an E step (responsibilities γ_ik = the posterior that point i came from component k, by Bayes) and an M step (φ_k = mean responsibility, μ_k and σ_k as responsibility-weighted statistics). EM guarantees the likelihood never decreases, converging to a local maximum. Fitted GMMs serve density estimation and soft clustering.

- Component weights φ_k sum to 1; each component has its own mean (and covariance in multivariate case).
- E step: γ_ik = φ_k N(x_i|μ_k, σ_k) / Σ_j φ_j N(x_j|μ_j, σ_j).
- M step: φ_k = Σγ/N, μ_k = Σγx/Σγ, σ_k² = Σγ(x−μ_k)²/Σγ.
- Likelihood strictly increases per iteration; result is a local maximum.
- Used for speech feature extraction, multi-object tracking, density estimation, clustering.

## Gaussian Mixture Model (GMM) using Expectation Maximization (EM) Technique (IIT Madras, based on Bishop's PRML)

URL: https://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf  ·  Date: unknown  ·  Status: summary

Lecture slides from the IIT Madras video processing lab. They open with the Gaussian density (univariate and multivariate) and the maximum-likelihood estimates of mean and covariance for a single Gaussian, then move to mixtures: p(x) = Σ π_k N(x | μ_k, Σ_k), a linear superposition of K Gaussians whose mixing coefficients satisfy 0 ≤ π_k ≤ 1 and sum to 1. The mixture log-likelihood has no closed-form maximum, so EM fits it: treat the mixing coefficients as priors, get the posterior responsibilities γ_k(x) by Bayes, and read N_k = Σ γ_k(x_n) as the effective point count of cluster k. The loop is initialize, E step (responsibilities), M step (weighted re-estimates of μ, Σ, π), evaluate the log-likelihood, repeat until convergence.

- Single-Gaussian ML: μ_ML and Σ_ML are the sample mean and biased sample covariance.
- Mixture: p(x) = Σ_k π_k N(x | μ_k, Σ_k), with π_k ≥ 0 and Σ π_k = 1.
- Responsibilities = posterior component assignments via Bayes' rule; N_k = effective points in cluster k.
- EM loop: init → E step (γ) → M step (μ, Σ, π updates) → log-likelihood check → repeat.

## What is Hierarchical Clustering? (Displayr — Tim Bock)

URL: https://www.displayr.com/what-is-hierarchical-clustering/  ·  Date: unknown  ·  Status: summary

A practical explainer. Hierarchical clustering takes raw data or a precomputed distance matrix, merges the two nearest clusters repeatedly until one cluster remains, and outputs a dendrogram whose merge heights encode similarity. Beyond the distance metric (Euclidean by default), the linkage criterion decides which points define cluster-to-cluster distance (single, complete, average, and others); with no domain reason otherwise, Ward's method is the sensible default because it minimizes the sum of squared distances of points from their cluster average, matching the ANOVA-style assumptions of standard statistics. In Python, scipy's linkage(method='ward'), dendrogram(Z), and fcluster cut the tree into k clusters. Limitations: slow on large data, sensitive to metric and linkage choices, and distorted by noise and outliers.

- Input: raw data or a distance matrix; clusters merge until one remains, hierarchy shown in a dendrogram.
- Linkage decides which points measure cluster distance; Ward's minimizes within-cluster sum of squares.
- Python: scipy linkage → dendrogram → fcluster(Z, t, criterion='maxclust'); R: hclust(dist(data), method="ward.D2").
- Drawbacks: high time/memory cost, sensitivity to choices, poor behavior with noise and outliers.

## Definitions to skim

- Cluster analysis — a family of algorithms that partition objects into groups so same-group objects are more similar to each other than to objects in other groups. https://en.wikipedia.org/wiki/Cluster_analysis
- K-means clustering — partitions n observations into K groups, each represented by the mean (centroid) of its points, minimizing within-cluster variance. https://en.wikipedia.org/wiki/K-means_clustering
- Multimodal distribution — a probability distribution with more than one mode (peak); a bimodal distribution is the two-peak case, often a mixture of two unimodal distributions. https://en.wikipedia.org/wiki/Multimodal_distribution
- Mixture model — a probabilistic model for subpopulations within a population, inferred from pooled observations without per-point subpopulation labels. https://en.wikipedia.org/wiki/Mixture_model
- Expectation-maximization algorithm — an iterative method for (local) maximum-likelihood parameter estimation in models with latent variables, alternating an expectation step and a maximization step. https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
- Hierarchical clustering — cluster analysis that builds a tree (hierarchy) of clusters, agglomerative (bottom-up merging) or divisive (top-down splitting). https://en.wikipedia.org/wiki/Hierarchical_clustering
- Ward's method — a minimum-variance criterion for agglomerative clustering: at each step merge the pair of clusters whose union increases total within-cluster variance the least. https://en.wikipedia.org/wiki/Ward%27s_method
- Cophenetic — the cophenetic distance of two objects is the dendrogram height at which their branches first merge into one. https://en.wikipedia.org/wiki/Cophenetic

## References

- scikit-learn — https://scikit-learn.org/stable/index.html
- Clustering (sklearn user guide) — https://scikit-learn.org/stable/modules/clustering.html
- sklearn.cluster.KMeans — https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Gaussian mixture models — https://scikit-learn.org/stable/modules/mixture.html
- sklearn.mixture.GaussianMixture — https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- scipy — https://scipy.org/
- scipy.cluster.hierarchy — https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
- scipy.cluster.hierarchy.linkage — https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
- scipy.cluster.hierarchy.fcluster — https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
- scipy.cluster.hierarchy.dendrogram — https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

## Quiz Hooks

- clustering — grouping data points so points in the same group are more alike than points in other groups; unsupervised learning.
- centroid — the center of a cluster; for k-means, the mean vector of the cluster's points.
- k-means — algorithm that splits data into K clusters, each represented by its centroid, by alternating assign-to-nearest and recompute-mean.
- WCSS/variance — within-cluster sum of squares: total squared distance from each point to its own centroid; the quantity k-means minimizes.
- elbow method — rule for choosing K: take the value where the WCSS-versus-K curve stops dropping steeply (maximum curvature / second derivative).
- bimodal/multimodal distribution — a distribution with two or more distinct peaks, usually a sign that several underlying groups produced the data.
- mixture model — a model of a population as a weighted sum of simpler distributions, learned without per-point labels.
- Gaussian Mixture Model — a mixture model whose components are Gaussian (normal) distributions, each with its own mean and covariance.
- expectation-maximization — iterative maximum-likelihood algorithm for latent-variable models: E step computes expected assignments, M step re-estimates parameters, repeat.
- responsibility — the posterior probability γ that a data point was generated by a given mixture component.
- likelihood — the probability of the observed data under the model's parameters; EM maximizes it.
- hierarchical clustering — clustering that produces a nested tree of clusters (a dendrogram) rather than a single flat partition.
- linkage — the rule defining the distance between two clusters from pairwise point distances (single = min, complete = max, average, Ward, ...).
- Ward's method — linkage that merges the pair of clusters causing the smallest increase in total within-cluster variance.
- dendrogram — tree diagram showing merge order and merge heights in hierarchical clustering.
- cophenetic distance — for two points, the height of the dendrogram node where their branches join; the dissimilarity at which they land in the same cluster.
