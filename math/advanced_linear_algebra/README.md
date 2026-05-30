# Math — Advanced Linear Algebra

A progressive journey from manually computing determinants with Python lists to classifying matrix definiteness with NumPy — building deep intuition for the linear algebra pipeline: determinant → minor → cofactor → adjugate → inverse → definiteness.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Compute the determinant of any square matrix recursively using Laplace expansion (no NumPy) |
| 2 | Build a matrix of minors by extracting submatrices and computing their determinants |
| 3 | Apply the checkerboard sign pattern $(-1)^{i+j}$ to convert minors into cofactors |
| 4 | Compute the adjugate (classical adjoint) as the transpose of the cofactor matrix |
| 5 | Find the inverse by dividing the adjugate by the determinant — handle singular matrices |
| 6 | Classify symmetric matrices by definiteness using NumPy eigenvalue sign analysis |
| 7 | Combine all concepts into a pipeline: det → minor → cofactor → adjugate → inverse |

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from earlier tasks are not repeated. Tasks 0–4 use **only Python lists** (no NumPy); Task 5 introduces NumPy.

---

### Task 0 — Determinant (`0-determinant.py`)

**Challenge:** Compute the determinant of a square matrix of arbitrary size using only Python lists — implementing Laplace expansion from scratch with proper input validation.

**Approach:** The function handles three base cases (0×0 → 1, 1×1 → the value, 2×2 → $ad - bc$) and recursively expands along the first row for larger matrices. For each column $j$, it builds the submatrix by skipping row 0 and column $j$, then accumulates $(-1)^j \cdot a_{0j} \cdot \det(\text{submatrix})$.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `matrix == [[]]` → return 1 | Base case: 0×0 matrix determinant is 1 |
| `ad - bc` for 2×2 | Direct formula avoids recursion overhead |
| `matrix[row][:col] + matrix[row][col+1:]` | Slice-based column removal (no `np.delete`) |
| `(-1) ** col` alternating sign | Laplace expansion sign pattern for first-row expansion |
| Recursive `determinant(submatrix)` | Recursively reduce problem to smaller submatrices |
| Input validation chain | `isinstance(list)` → `all(isinstance(row, list))` → square check |

> **Key takeaway:** The determinant is the foundation of the entire linear algebra pipeline. Laplace expansion recursively breaks an n×n problem into n problems of size (n−1)×(n−1) — pure divide and conquer.

---

### Task 1 — Matrix of Minors (`1-minor.py`)

**Challenge:** Build a full matrix of minors — for every position $(i, j)$, compute the determinant of the submatrix formed by deleting row $i$ and column $j$.

**Approach:** Reuses `determinant()` from Task 0. Implements special-case shortcuts for 1×1 (`[[1]]`) and 2×2 (reverse rows and columns). For 3×3+, iterates over all $(i, j)$ pairs, builds each submatrix by skipping row $i$ and column $j$, computes its determinant, then reassembles results into the minor matrix shape using list slicing `temp[i*n:(i+1)*n]`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `continue` to skip row | Cleanly omit a row from submatrix construction |
| `for k in range(n): if k == i: continue` | Build submatrix excluding one specific row |
| Row-slicing reassembly `temp[i*n:(i+1)*n]` | Convert flat list of minor values into 2D matrix rows |
| 2×2 minor shortcut `matrix[row][::-1]` | Direct formula avoids submatrix/det overhead for small case |

> **Key takeaway:** The minor matrix generalizes the determinant — instead of one number, you compute a determinant for every position $(i, j)$. Each minor tells you "how much does this position matter?"

---

### Task 2 — Cofactor Matrix (`2-cofactor.py`)

**Challenge:** Convert the matrix of minors into a cofactor matrix by applying the checkerboard sign pattern $(-1)^{i+j}$.

**Approach:** Calls `minor()` from Task 1, then iterates over every element multiplying by $(-1)^{\text{row}+\text{col}}$. The sign pattern creates alternating signs like a chess board: $+$ at even $i+j$, $-$ at odd $i+j$.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `(-1) ** (row + col)` | Checkerboard sign pattern — positive at even positions, negative at odd |
| `cofactor = sign × minor` | Mathematical relationship: $C_{ij} = (-1)^{i+j} M_{ij}$ |
| Cumulative task dependency | Task 2 imports `minor()` from Task 1, which uses `determinant()` from Task 0 |

> **Key takeaway:** The cofactor is just a signed minor — same values, alternating signs. The sign pattern is what turns minors into the building blocks of inverses.

---

### Task 3 — Adjugate Matrix (`3-adjugate.py`)

**Challenge:** Compute the adjugate (classical adjoint) — the transpose of the cofactor matrix. This is the final preprocessing step before the inverse.

**Approach:** Calls `cofactor()` from Task 2, then transposes by iterating with `cofact[col][row]` (swapped indices). Reassembles into 2D using row slicing.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `cofact[col][row]` (swapped indices) | Manual matrix transpose — iterates columns-first, rows-second |
| Adjugate = transpose of cofactor | $\text{adj}(A) = C^T$ — swap rows and columns of the cofactor matrix |
| Full pipeline dependency | `adjugate()` → `cofactor()` → `minor()` → `determinant()` |

> **Key takeaway:** The adjugate is the transpose of the cofactor matrix — a simple operation that completes the preprocessing chain. The inverse is now just one division away: $A^{-1} = \text{adj}(A) / \det(A)$.

---

### Task 4 — Inverse (`4-inverse.py`)

**Challenge:** Compute the matrix inverse by dividing the adjugate by the determinant — and detect singular (non-invertible) matrices.

**Approach:** Calls `adjugate()` and `determinant()`. If $\det = 0$, returns `None` (singular). Otherwise divides every element of the adjugate by the determinant: `adju[row][col] / det`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `det == 0` → return `None` | Singular matrix detection — no inverse exists |
| `adju[row][col] / det` | Inverse formula: $A^{-1} = \frac{1}{\det(A)} \cdot \text{adj}(A)$ |
| Scalar division of every element | Building a new matrix by scaling each entry |
| Full pipeline: det → minor → cofactor → adjugate → inverse | Complete linear algebra chain, all from Python lists |

> **Key takeaway:** The inverse is the culmination of the entire pipeline — every previous step (determinant, minor, cofactor, adjugate) was building toward this. If $\det = 0$, the matrix is singular and has no inverse.

---

### Task 5 — Definiteness (`5-definiteness.py`)

**Challenge:** Classify a symmetric matrix by its definiteness (positive/negative definite, semi-definite, or indefinite) — introducing NumPy and eigenvalue analysis.

**Approach:** Validates input as `np.ndarray`, checks symmetry with `np.allclose(matrix, matrix.T)`, then uses `np.linalg.eigvalsh()` to get eigenvalues. Classifies by sign: all $> 0$ → positive definite, all $\geq 0$ → positive semi-definite, all $< 0$ → negative definite, all $\leq 0$ → negative semi-definite, mixed → indefinite.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `isinstance(matrix, np.ndarray)` | Require NumPy array input (vs Python lists in Tasks 0–4) |
| `np.allclose(matrix, matrix.T)` | Check matrix symmetry with floating-point tolerance |
| `np.linalg.eigvalsh(A)` | Efficient eigenvalue computation for symmetric/Hermitian matrices |
| `np.all(eigvals > tol)` | Classify definiteness by eigenvalue sign analysis |
| Five-way classification | Positive/negative definite, semi-definite, indefinite |
| Tolerance `tol = 1e-10` | Handle floating-point noise near zero |

> **Key takeaway:** Definiteness tells you the "shape" of a matrix — eigenvalues are the key. All positive eigenvalues = positive definite (bowl-shaped), mixed signs = indefinite (saddle-shaped). This concept is critical for optimization and ML.

---

### Bonus — Matrix Quiz (`quiz.py`)

**Challenge:** A comprehensive 10-question quiz that tests all six concepts using NumPy's `linalg` module as a verification tool — applying the theory to concrete 3×3 matrices.

**Approach:** Implements NumPy-based helper functions (`minor_matrix`, `cofactor_matrix`, `adjugate_matrix`, `classify_definiteness`) and runs each concept against multiple-choice options using `np.allclose()`. Also explores eigenvalue power iteration ($A^{10}v = \lambda^{10}v$) and eigenpair verification.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.linalg.det()` / `np.linalg.inv()` / `np.linalg.eig()` | NumPy verification of manual implementations |
| `np.allclose(result, option)` | Match computed results to multiple-choice answers |
| Eigenvalue power iteration | $A^{10}v = \lambda^{10}v$ when $v$ is an eigenvector |
| `np.arange()[:, None]` broadcasting | Efficient sign pattern generation without loops |
| `np.delete(A, i, axis=0)` | NumPy alternative to the manual submatrix construction |

> **Key takeaway:** The quiz validates your manual implementations against NumPy's battle-tested `linalg` module — and shows how eigenvalues unlock efficient matrix powers without computing $A^{10}$ directly.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | Recursive Laplace expansion, 2×2 shortcut, `(-1)^col` sign | Determinant (Manual) |
| 1 | Submatrix by `continue` skip, flat-to-2D reassembly | Minors (Manual) |
| 2 | Checkerboard $(-1)^{row+col}$ sign on minors | Cofactors (Manual) |
| 3 | Manual transpose `matrix[col][row]` of cofactor | Adjugate (Manual) |
| 4 | Inverse = adjugate / det, singular → `None` | Inverse (Manual) |
| 5 | `np.linalg.eigvalsh()`, symmetry check, definiteness classification | Definiteness (NumPy) |
| quiz | `np.linalg.det/inv/eig`, power iteration, `np.delete()` | Verification (NumPy) |

---

## Resources

- [NumPy linalg.det — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)
- [NumPy linalg.eigvalsh — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html)
- [NumPy linalg.inv — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)
- [Matrix Minor, Cofactor, Adjugate — Wikipedia](https://en.wikipedia.org/wiki/Minor_(linear_algebra))
- [Definiteness of a Matrix — Wikipedia](https://en.wikipedia.org/wiki/Definite_matrix)
- [Laplace Expansion — Wikipedia](https://en.wikipedia.org/wiki/Laplace_expansion)
