# Math — Advanced Linear Algebra

Advanced matrix concepts beyond basic operations — determinants, minors, cofactors, adjugates, inverses, eigenvalues, eigenvectors, and matrix definiteness classification.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Compute the determinant of a square matrix using NumPy |
| 2 | Build a matrix of minors by extracting submatrices and computing their determinants |
| 3 | Apply the checkerboard sign pattern to convert minors into cofactors |
| 4 | Compute the adjugate (classical adjoint) as the transpose of the cofactor matrix |
| 5 | Find the inverse of a matrix and detect singular (non-invertible) matrices |
| 6 | Use eigenvalues and eigenvectors for efficient matrix power computation |
| 7 | Verify candidate eigenvalue/eigenvector pairs programmatically |
| 8 | Classify symmetric matrices by definiteness using eigenvalue sign analysis |

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from linear algebra (basic NumPy operations) are assumed and not repeated.

---

### Task 0 — Matrix Quiz (`quiz.py`)

**Challenge:** A comprehensive 10-question quiz spanning the full pipeline of advanced matrix analysis — from raw determinant computation through minors, cofactors, adjugates, and inverses, culminating in eigenvalue analysis and definiteness classification. Each question builds on the concepts of the previous ones.

**Approach:** The quiz implements three helper functions — `minor_matrix()`, `cofactor_matrix()`, and `adjugate_matrix()` — then applies them together with NumPy's `linalg` module. Each question isolates a specific concept:

- **Q0:** `np.linalg.det(A)` — determinant of a 3×3 matrix
- **Q1:** `minor_matrix()` — iterate over every (i, j) position, delete row i and column j with `np.delete()`, then compute `np.linalg.det()` of the resulting submatrix
- **Q2:** `cofactor_matrix()` — multiply each minor by $(-1)^{i+j}$ (checkerboard sign pattern applied via `np.arange()` broadcasting)
- **Q3:** `adjugate_matrix()` — transpose the cofactor matrix (`.T`)
- **Q4:** `np.linalg.inv(A)` — compute the inverse; verify by checking `np.allclose()` against given options
- **Q5:** Singularity detection — `np.isclose(det, 0)` identifies non-invertible matrices
- **Q6:** Eigenvalue power iteration — if $v$ is an eigenvector, $A^{10}v = \lambda^{10}v$; extract $\lambda$ from $Av / v$, then compute $\lambda^{10}v$ without computing $A^{10}$
- **Q7:** Eigenpair verification — $A @ v$ should equal $\lambda v$ for a valid eigenpair; `np.linalg.eig()` returns all eigenvalues/vectors
- **Q8–Q10:** `classify_definiteness()` — use `np.linalg.eigvalsh()` for symmetric matrices, then check eigenvalue signs: all $>0$ → positive definite, all $\geq 0$ → positive semi-definite, all $<0$ → negative definite, all $\leq 0$ → negative semi-definite, mixed signs → indefinite

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.linalg.det(A)` | Compute the determinant of a square matrix |
| `np.delete(A, i, axis=0)` | Remove row i from a 2D array |
| `np.delete(np.delete(A, i, axis=0), j, axis=1)` | Extract submatrix by deleting one row and one column |
| `(-1) ** (np.arange(n)[:, None] + np.arange(n))` | Generate checkerboard sign pattern for cofactor computation |
| `.T` (transpose property) | Flip cofactor matrix to produce the adjugate |
| `np.linalg.inv(A)` | Compute matrix inverse (raises error if singular) |
| `np.isclose(det, 0)` / `np.isclose(A, B)` | Floating-point-safe equality checks |
| `np.linalg.eig(A)` | Return eigenvalues and eigenvectors of a general matrix |
| `np.linalg.eigvalsh(A)` | Return eigenvalues of a symmetric/Hermitian matrix (more stable) |
| `A @ v` (matrix-vector product) | Verify eigenpair: $Av = \lambda v$ |
| $\lambda^{10}v$ (power iteration) | Compute $A^{10}v$ in O(1) when $v$ is an eigenvector |
| `np.allclose(arr, val, tol)` | Check if all elements are close to a value within tolerance |

> **Key takeaway:** The determinant pipeline — minors → cofactors → adjugate → inverse — is a unified process where each step builds on the last. Eigenvalues unlock efficient matrix powers and reveal the fundamental nature of a matrix through definiteness classification.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0  | `np.linalg.det()` — determinant computation | Determinants |
| 0  | `np.delete()` — submatrix extraction via row/col removal | Submatrices |
| 0  | Minor matrix — iterate (i,j), delete row i col j, det of remainder | Minors & Cofactors |
| 0  | Cofactor sign pattern — $(-1)^{i+j}$ applied to minors | Minors & Cofactors |
| 0  | Adjugate — transpose of cofactor matrix | Inverses |
| 0  | `np.linalg.inv()` — matrix inverse | Inverses |
| 0  | `np.isclose()` — safe floating-point comparison | Utilities |
| 0  | Singular matrix detection — det ≈ 0 | Inverses |
| 0  | Eigenvalue power iteration — $A^{10}v = \lambda^{10}v$ | Eigenvalues |
| 0  | `np.linalg.eig()` — eigenvalue/vector computation | Eigenvalues |
| 0  | Eigenpair verification — $Av = \lambda v$ | Eigenvalues |
| 0  | `np.linalg.eigvalsh()` — symmetric eigenvalue computation | Eigenvalues |
| 0  | Definiteness classification — eigenvalue sign analysis | Definiteness |

---

## Resources

- [NumPy linalg.det — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)
- [NumPy linalg.inv — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)
- [NumPy linalg.eig — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)
- [NumPy linalg.eigvalsh — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html)
- [NumPy delete — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.delete.html)
- [Matrix Minor, Cofactor, Adjugate — Wikipedia](https://en.wikipedia.org/wiki/Minor_(linear_algebra))
- [Definiteness of a Matrix — Wikipedia](https://en.wikipedia.org/wiki/Definite_matrix)
