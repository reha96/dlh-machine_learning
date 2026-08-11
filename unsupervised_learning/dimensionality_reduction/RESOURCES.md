# Dimensionality Reduction — Resources

Intranet project: PCA (tasks 0-1) + t-SNE (tasks 2-8), numpy only, no
sklearn. Verification env: `my-venv` (numpy 2.4.4). Data: mnist2500_X.txt.

## Read or watch

### Dimensionality Reduction For Dummies — Part 1: Intuition (Hussein Abdul)

URL: https://medium.com/data-science/https-medium-com-abdullatif-h-dimensionality-reduction-for-dummies-part-1-a8c9ec7b7e79  ·  Date: 2018-10-18 (publication date)  ·  Status: paywalled

Medium member-only story; no Wayback Machine snapshot exists, and the page
serves only the teaser even in a browser. It opens: "Humans are visual
creatures. We need to see in order to believe." — with more than three
dimensions the data becomes impossible to see, so the article asks whether
the extra dimensions are really necessary. PCA is named as one simple,
elegant answer, then a house-value dataset (4 features per item) is meant to
carry the intuition; the rest is behind the paywall.

- Only the intro paragraphs are visible; the working example is locked.
- Theme: reduce data to 1-3 "humanly" dimensions while keeping information.

### Understanding SVD (Singular Value Decomposition) (Gregory Gundersen)

URL: https://gregorygundersen.com/blog/2018/12/10/svd/  ·  Date: 2018-12-10 (publication date)  ·  Status: summary

Builds the SVD up from first principles instead of stating it. Geometric
essence: any linear transformation of a square can be seen as just
stretching, compressing, or flipping — provided you are allowed to rotate
the square before and after. The singular values are the side lengths of
the transformed shape; the largest one is the maximum "action" of the
matrix. Formalizes this to M = UΣV*, with orthonormal left/right singular
vectors in U and V and ordered, nonnegative singular values on Σ's diagonal.
Closes with PCA: since X^T X = VΣ²V^T, PCA is exactly diagonalizing the
covariance matrix.

- SVD = rotate, scale (Σ), rotate back; any matrix admits it.
- Singular values = how much the map stretches each principal direction.
- Number of nonzero singular values = rank; a zero value flattens a dimension.
- X^T X = VΣ²V^T — this is why PCA and SVD are the same computation.

### Intuitively, what is the difference between Eigendecomposition and SVD? (Math StackExchange)

URL: https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu  ·  Date: 2013-03-04 (publication date)  ·  Status: summary

Top-voted answers frame the difference conceptually. Eigendecomposition
A = PDP⁻¹ describes a transformation using a basis P that need not be
orthonormal, and exists only for square matrices — and even then not always
(no real eigenvalues). SVD A = UΣV* always exists, uses orthonormal U and V
(so pure rotations), and the diagonal entries are real and nonnegative.
An accepted answer puts it sharply: eigendecomposition finds directions a
matrix does not rotate; SVD finds corresponding vectors that are scaled the
same whether moving from left space to right space or vice-versa. A 2D
rotation matrix has no eigenvectors at all, yet a trivial SVD (scale by 1).

- SVD works on any rectangular matrix; eigendecomposition needs a square one.
- U and V are orthonormal; P and P⁻¹ are inverses but generally not rotations.
- Σ entries are real, nonnegative; D's eigenvalues may be complex.
- Rotation in R²: no real eigenpairs, but a clean SVD.

### How to Use t-SNE Effectively (Distill, Wattenberg, Viégas, Johnson)

URL: https://distill.pub/2016/misread-tsne/  ·  Date: 2016-10-13 (publication date)  ·  Status: summary

Interactive essay from Google Brain on reading t-SNE plots without
misreading them. Main lessons: hyperparameters matter — iterate until the
configuration is stable, and keep perplexity in the suggested 5-50 range
but below the number of points. Cluster sizes in the plot mean nothing,
because t-SNE deliberately expands dense clusters and contracts sparse
ones. Distances between well-separated clusters are also unreliable: seeing
global geometry requires fine-tuning a single global perplexity. Pure
random noise can look clustered, especially at low perplexity. Some shapes
and topology do come through, but usually only at multiple perplexities and
across several runs.

- Perplexity balances local vs global attention; too small → local noise dominates.
- Density equalization is by design: sizes are not informative.
- Inter-cluster distances may be meaningless; no single perplexity fits all data.
- Low perplexity on 100-dim Gaussian noise shows convincing fake clusters.
- Pinched or 1-D-looking layouts often mean the run was stopped too early.

### Singular Value Decomposition (Artificial Intelligence - All in One)

URL: https://www.youtube.com/watch?v=P5mlg91as1c  ·  Date: 2016-04-13 (YouTube upload)  ·  Status: failed-transcript

Captions are unavailable for this video (NoTranscriptFound), so no summary
can be written from it.

### PCA Part 1 (Data4Bio)

URL: https://www.youtube.com/watch?v=ZqXnPcyIAL8  ·  Date: 2016-06-10 (YouTube upload)  ·  Status: summary (transcript)

PCA is the eigendecomposition of the covariance matrix X^T X, where X holds
n samples and m measurements. The eigenvector matrix W (the "loadings") is
m×m; its columns are the principal components, ordered by decreasing
eigenvalue, so the first component explains the most variance. The
projected data ("scores") T = XW is just the data described in a new basis:
same shape, same relative distances, only rotated. The payoff is
truncation — keep the first r columns of W and get an n×r view, with the
first component pointing along the main axis of the data ellipsoid.

- Columns of W = principal components, ordered by eigenvalue size.
- Scores T = XW: a rotation, not a change of the data.
- Truncate W to its first r columns for the best r-dimensional view.
- PC1 aligns with the longest axis of the (elliptical) point cloud.

### PCA Part 2 (Data4Bio)

URL: https://www.youtube.com/watch?v=NUn6WeFM5cM  ·  Date: 2016-06-10 (YouTube upload)  ·  Status: summary (transcript)

PCA is so popular it was reinvented many times; the SVD is its efficient
incarnation. Decompose X = UΣV*: U holds left singular vectors, V right
singular vectors (identical to PCA's W), and Σ is diagonal with ordered,
nonnegative singular values. Since X^T X = VΣ²V^T, SVD skips building the
covariance matrix and is faster. U and V are unitary (V*V = I), and the
scores fall out for free: T = XV = UΣ. To choose how many components to
keep, plot the cumulative sum of singular values — a sharp elbow means
truncate there; a shallow curve means no clear break, and a common
convention is to keep enough components for 95% of the variance.

- V from the SVD equals W from the covariance eigendecomposition.
- Σ diagonal holds singular values, ordered σ1 ≥ σ2 ≥ ... ≥ 0.
- Scores T = UΣ come directly from the decomposition.
- Pick r at the elbow of the cumulative-singular-value curve, or at 95% variance.

### StatQuest: t-SNE, Clearly Explained (Josh Starmer)

URL: https://www.youtube.com/watch?v=NEaUSP4YerM  ·  Date: 2017-09-18 (YouTube upload)  ·  Status: summary (transcript)

Walks through t-SNE on a toy 2D scatter plot reduced to a 1D number line:
points are placed randomly, then each point is attracted to the points it
sits near in the original space and repelled by those it sits far from,
moving a little at a time. Similarity in the high-dimensional space is the
height of a normal curve centered on the point; scores are scaled so each
row sums to 1, which equalizes dense and sparse clusters (wider curve for
sparser regions). The low-dimensional map uses a t distribution instead —
flatter in the middle, fatter tails — and that is the "t" in t-SNE; without
it the clusters all clump in the middle and are hard to see. Perplexity
enters as the expected density around each point.

- Attraction and repulsion, moved step by step, produces the clusters.
- Scaling similarities to sum to 1 makes dense and sparse clusters comparable.
- Perplexity = expected number of neighbors around each point.
- The t distribution's heavy tails stop everything from piling into the center.

### t-SNE tutorial Part 1 (Divy Kangeyan)

URL: https://www.youtube.com/watch?v=ohQXphVSEQM  ·  Date: 2017-05-27 (YouTube upload)  ·  Status: summary (transcript)

t-SNE (van der Maaten and Hinton, 2008) projects high-dimensional data to
2 or 3 dimensions for visualization while preserving local structure, where
PCA and MDS only keep global structure. The precursor SNE builds
conditional probabilities p_{j|i} from a Gaussian kernel of the distances
(close points → high probability) and q_{j|i} in the low-dimensional space
with variance fixed at 1/√2. The cost function is the Kullback-Leibler
divergence between the two distributions, minimized by gradient descent
with a momentum term. The per-point variance σ_i is set by a parameter
called perplexity (roughly the number of neighbors) via binary search on
the entropy. SNE's drawbacks: hard optimization and the "crowding problem"
— moderately distant points all get clumped together.

- p_{j|i} close to 1 for near pairs, ~0 for far pairs (Gaussian kernel).
- Cost = KL divergence between P and Q; optimize by gradient descent + momentum.
- Perplexity fixes σ_i per point through a binary search.
- Crowding problem and tricky optimization are what t-SNE was built to fix.

### t-SNE tutorial Part 2 (Divy Kangeyan)

URL: https://www.youtube.com/watch?v=W-9L6v_rFIE  ·  Date: 2017-05-27 (YouTube upload)  ·  Status: summary (transcript)

The two fixes from Part 1 in detail. First, symmetric SNE: p_ij =
(p_{i|j} + p_{j|i}) / 2n, with the diagonal set to zero. Second, the low
dimensional map uses a student-t distribution with one degree of freedom
(the Cauchy), which has no exponential term — cheaper to evaluate and
robust to outliers. The algorithm: compute affinities, seed Y from
N(0, 10⁻⁴), then loop: recompute Q, take the KL gradient, apply the
gradient-descent update with momentum. Implementations exist in R (Rtsne),
Python (sklearn.manifold.TSNE), Julia, and MATLAB. Ends with the critiques
of Wattenberg et al.: perplexity changes cluster structure, cluster size
and inter-cluster distance carry no meaning, and random noise can produce
convincing but false structure.

- Symmetric affinities p_ij = (p_{i|j} + p_{j|i}) / 2n; diagonal zero.
- Cauchy (t with 1 dof) in low dimensions: no exponent, faster, outlier-robust.
- Y starts at N(0, 10⁻⁴); iterate Q → gradient → update until convergence.
- t-SNE axes and distances are not interpretable like PCA's.

## Definitions to skim

- Dimensionality reduction — mapping high-dimensional data to a lower-dimensional representation that keeps its essential structure — https://en.wikipedia.org/wiki/Dimensionality_reduction (last edited 2026-07-27)
- Principal component analysis — orthogonal projection of the data onto the directions of maximum variance — https://en.wikipedia.org/wiki/Principal_component_analysis (last edited 2026-07-13)
- Eigendecomposition of a matrix — factorizing a square matrix as PDP⁻¹, with eigenvectors in P and eigenvalues on the diagonal of D — https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix (last edited 2026-08-06)
- Singular value decomposition — factorizing any matrix as UΣV*, with unitary U and V and a diagonal Σ of nonnegative singular values — https://en.wikipedia.org/wiki/Singular_value_decomposition (last edited 2026-07-22)
- Manifold — a topological space that locally resembles Euclidean space; the shape high-dimensional data is assumed to lie on — https://en.wikipedia.org/wiki/Manifold (last edited 2026-07-07)
- Kullback-Leibler divergence — an asymmetric measure of how one probability distribution differs from another — https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence (last edited 2026-07-15)
- T-distributed stochastic neighbor embedding — t-SNE; embeds high-dimensional points in a low-dimensional map by matching pairwise similarity distributions (Gaussian in high dim, student-t in low dim) — https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding (last edited 2026-01-14)

## References (link only)

- numpy.cumsum — https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
- Visualizing Data using t-SNE (JMLR paper PDF) — https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
- Visualizing Data Using t-SNE (video) — https://www.youtube.com/watch?v=RJVL80Gg3lA
- Kernel principal component analysis — https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
- Nonlinear Dimensionality Reduction: KPCA (video) — https://www.youtube.com/watch?v=HbDHohXPLnU

## Quiz Hooks

- PCA — finds the orthogonal directions of maximum variance in the data.
- Eigendecomposition — A = PDP⁻¹; square matrices only, basis need not be orthonormal.
- SVD — X = UΣV*; exists for any matrix; U and V are orthonormal.
- Singular values — diagonal entries of Σ; nonnegative, ordered; square roots of the eigenvalues of X^T X.
- Explained variance — the share of total variance carried by a component (λ_i / Σλ).
- Cumulative variance — running sum of explained variance; used to pick r (e.g. keep 95%).
- Manifold — the locally Euclidean surface the high-dimensional data is assumed to lie on.
- t-SNE — nonlinear embedding matching Gaussian high-dim similarities to student-t low-dim similarities via KL divergence.
- Perplexity — the effective number of neighbors per point; sets σ_i in the high-dim Gaussian.
- KL divergence — asymmetric cost; heavily penalizes mapping close points far apart, lightly the reverse.
- Low-dimensional embedding — the coordinates Y learned by t-SNE; no intrinsic axis meaning.
- Whitening — transforming data so features are uncorrelated with unit variance.
- KPCA — the kernel trick applied to PCA, for structure that is nonlinear in the input space.
