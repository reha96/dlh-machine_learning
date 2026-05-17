# Math — Linear Algebra

A progressive journey from manual matrix operations using Python lists to vectorized NumPy computations — building intuition for the linear algebra foundations of machine learning.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Extract sub-arrays and sub-matrices using Python slicing notation |
| 2 | Compute matrix shape, transpose, and perform element-wise operations manually |
| 3 | Perform matrix addition, concatenation, and multiplication with nested loops |
| 4 | Transition from manual Python list operations to NumPy's vectorized API |
| 5 | Use NumPy for slicing, transposing, broadcasting, and matrix multiplication |
| 6 | Generalize matrix operations to n-dimensional arrays with recursion |

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from earlier tasks are not repeated.

---

### Task 0 — Basic Slicing (`0-slice_me_up.py`)

**Challenge:** Extract different portions of a 1D array using Python's slicing notation — introducing the fundamental indexing syntax.

**Approach:** `arr[:2]` for first two elements, `arr[-5:]` for last five, `arr[1:6]` for a middle range. String formatting with `.format()` prints results.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `list[start:end]` | Extract a contiguous sub-list by index range |
| Negative indexing (`-5:`) | Count from the end — `-1` is the last element |
| `"{}".format(value)` | String interpolation for printing results |

> **Key takeaway:** Python's slice notation `[start:stop]` is concise and readable — `start` is inclusive, `stop` is exclusive.

---

### Task 1 — 2D Row Slicing (`1-trim_me_down.py`)

**Challenge:** Extract middle columns from every row of a 2D matrix — combining row iteration with column slicing.

**Approach:** Loop through each row, slice columns 2–4 with `row[2:6]`, append to a result list.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Nested list iteration | Iterate through a 2D structure row-by-row |
| Row-level slicing in a loop | Apply the same slice to every row |
| `list.append()` for result building | Accumulate processed rows into a new matrix |

> **Key takeaway:** Combining row iteration with slicing extracts multi-dimensional submatrices in a single pass.

---

### Task 2 — Matrix Shape (`2-size_me_please.py`)

**Challenge:** Calculate all dimensions (shape) of a nested list matrix of arbitrary depth — introducing recursive type inspection.

**Approach:** While the current level is a list, record its length and descend into `mat[0]`. Uses `isinstance()` to detect list vs. scalar.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `isinstance(obj, list)` | Check if an object is a list (vs. scalar) |
| While-loop recursive descent | Drill into nested structures without explicit recursion |
| Shape as a list of lengths | Return `[rows, cols, ...]` for arbitrary dimensions |

> **Key takeaway:** Recursive dimension inspection with `isinstance` handles arbitrary nesting levels — the shape of a scalar is `[]`.

---

### Task 3 — Matrix Transpose (`3-flip_me_over.py`)

**Challenge:** Return the transpose of a 2D matrix (swap rows and columns) using only nested loops.

**Approach:** Outer loop over column indices, inner loop over row indices — building `result[j][i] = matrix[i][j]`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `range(len(matrix[0]))` for column count | Iterate over columns, not rows |
| Coordinate swap `[i][j] → [j][i]` | The core transpose operation |
| Pre-allocated result matrix | Build the output structure before filling |

> **Key takeaway:** Manual transposition systematically swaps row/column coordinates through nested loop iteration.

---

### Task 4 — 1D Element-wise Addition (`4-line_up.py`)

**Challenge:** Add two 1D arrays element-wise with shape validation — introducing operation preconditions.

**Approach:** Check `len(arr1) == len(arr2)`, return `None` if mismatch. Otherwise loop through indices and add corresponding elements.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `len(arr1) != len(arr2)` guard | Validate compatible dimensions before operating |
| Return `None` for invalid input | Signal failure without raising an exception |
| Element-wise addition with `for i in range(len(arr1))` | Add arrays index by index |

> **Key takeaway:** Always validate that arrays have matching shapes before performing element-wise operations.

---

### Task 5 — 2D Element-wise Addition (`5-across_the_planes.py`)

**Challenge:** Add two 2D matrices element-wise — extending validation and operations to two dimensions.

**Approach:** Validate both row count AND column count match. Use list comprehension `[[0] * len(arr1[0]) for _ in range(len(arr1))]` to pre-allocate the result matrix.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `[[0] * cols for _ in range(rows)]` | Pre-allocate a 2D list with list comprehension |
| 2D shape validation (rows + cols) | Both dimensions must match for element-wise ops |

> **Key takeaway:** 2D operations require validating both row and column dimensions before computation.

---

### Task 6 — 1D Concatenation (`6-howdy_partner.py`)

**Challenge:** Concatenate two 1D arrays into a single array — introducing bulk list extension.

**Approach:** Create an empty list, extend with `arr1`, then extend with `arr2`. Returns the combined array.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `list.extend(iterable)` | Add all elements from another iterable in one operation |
| Sequential concatenation | Preserve order: arr1 elements come first, then arr2 |

> **Key takeaway:** `extend()` efficiently concatenates arrays while maintaining element order — unlike `append()` which adds the whole list as one element.

---

### Task 7 — 2D Concatenation with Axis (`7-gettin_cozy.py`)

**Challenge:** Concatenate two 2D matrices along a specified axis (axis=0 for rows, axis=1 for columns).

**Approach:** For axis=0, use list addition `mat1 + mat2`. For axis=1, iterate rows and append columns from mat2 to each row of mat1.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Axis parameter for direction control | Choose row-wise (0) or column-wise (1) concatenation |
| `mat1 + mat2` for axis=0 | Direct list addition appends rows |
| Nested loop for axis=1 | Append column values row-by-row |

> **Key takeaway:** Matrix concatenation requires axis-dependent validation and merging strategies — axis=0 adds rows, axis=1 adds columns.

---

### Task 8 — Matrix Multiplication (`8-ridin_bareback.py`)

**Challenge:** Perform matrix multiplication on two 2D matrices — the most algorithmically complex manual operation in the series.

**Approach:** Validate that `cols(mat1) == rows(mat2)`. Triple-nested loop: outer for rows of mat1, middle for cols of mat2, inner for the dot product across the shared dimension.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Inner dimension validation | `cols(A) == rows(B)` must hold for multiplication |
| Triple-nested loop `(i, j, k)` | One loop per output coordinate pair + one for dot product |
| Dot product accumulation `sum = 0; sum += A[i][k] * B[k][j]` | Multiply and accumulate along the shared dimension |

> **Key takeaway:** Matrix multiplication requires three nested loops — the innermost `k` loop computes the dot product, the outer two iterate over output positions.

---

### Task 9 — NumPy: Slice and Transpose (`9-let_the_butcher_slice_it.py`)

**Challenge:** The first NumPy task — extract specific regions from a 4×6 array using NumPy slicing and `transpose()`.

**Approach:** `np.array()` creates the matrix. `np.transpose()` swaps dimensions to turn column extraction into row extraction, then slices normally. Combines transpose + slice for flexible extraction.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `import numpy as np` | Import NumPy with the standard alias |
| `np.array([[row1], [row2], ...])` | Create a NumPy array from nested Python lists |
| `np.transpose(arr)` | Swap rows and columns (or any axes in n-D) |

> **Key takeaway:** NumPy's `transpose()` turns column operations into row operations — slice after transpose to extract columns cleanly.

---

### Task 10 — NumPy Shape (`10-ill_use_my_scale.py`)

**Challenge:** Return the shape of a NumPy array without using loops or conditionals — leveraging NumPy's built-in properties.

**Approach:** Simply access and return `matrix.shape`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `array.shape` | NumPy property returning a tuple of dimension sizes |
| No-loop constraint | Demonstrate that built-in attributes eliminate manual iteration |

> **Key takeaway:** NumPy's `.shape` property directly provides array dimensions — no manual recursion or looping needed.

---

### Task 11 — NumPy Transpose (`11-the_western_exchange.py`)

**Challenge:** Transpose a NumPy array without loops or conditionals — NumPy's one-liner equivalent of Task 3.

**Approach:** Call `matrix.transpose()` and return the result.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `array.transpose()` | NumPy method — swaps dimensions in a single call |
| Method chaining | `.transpose()` returns a new array, enabling clean functional style |

> **Key takeaway:** What took nested loops in pure Python is a single method call in NumPy — `.transpose()` is both faster and more readable.

---

### Task 12 — Vectorized Element-wise Operations (`12-bracin_the_elements.py`)

**Challenge:** Perform four element-wise operations (add, subtract, multiply, divide) on NumPy arrays without loops.

**Approach:** Use NumPy's overloaded operators: `arr1 + arr2`, `arr1 - arr2`, `arr1 * arr2`, `arr1 / arr2`. Return results as a tuple.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| NumPy vectorized operators (`+`, `-`, `*`, `/`) | Apply operations to all elements simultaneously (broadcasting) |
| Tuple packing for multiple return values | `return (add, sub, mul, div)` |

> **Key takeaway:** NumPy's vectorized operators perform element-wise operations across all elements without explicit Python loops — this is the core of NumPy's speed advantage.

---

### Task 13 — NumPy Concatenation (`13-cats_got_your_tongue.py`)

**Challenge:** Concatenate two NumPy arrays along a specified axis — NumPy's equivalent of Task 7.

**Approach:** `np.concatenate((arr1, arr2), axis=axis)` — pass arrays as a tuple, specify the join axis.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.concatenate((a, b), axis=N)` | Join arrays along a given axis in one function call |
| Tuple argument for multiple arrays | `(arr1, arr2)` — concatenate more than two by adding to the tuple |

> **Key takeaway:** `np.concatenate()` efficiently joins arrays along any axis — a single call replaces the conditional logic of manual concatenation.

---

### Task 14 — NumPy Matrix Multiplication (`14-saddle_up.py`)

**Challenge:** Multiply two matrices using NumPy — the clean, mathematical equivalent of Task 8's triple loop.

**Approach:** Use the `@` operator (PEP 465): `mat1 @ mat2`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `@` operator (PEP 465) | Matrix multiplication operator — `A @ B` is equivalent to `np.matmul(A, B)` |
| Mathematical notation in code | `@` reads like the math: **A** × **B** |

> **Key takeaway:** NumPy's `@` operator makes matrix multiplication as clean as it looks on paper — `A @ B` replaces 10+ lines of nested loops.

---

### Task 100 — Multidimensional Slicing (`100-slice_like_a_ninja.py`)

**Challenge:** Slice a NumPy array along multiple axes using a dictionary specification — introducing `slice` objects and tuple-based indexing.

**Approach:** Create a list of `slice(None)` (one per dimension). Populate specific axes from the dictionary. Convert to tuple for indexing: `matrix[tuple(slices)]`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `slice(start, stop)` object | A parametrized slice that can be stored and passed around |
| `array.ndim` | Number of dimensions in the array |
| Tuple conversion for multi-dimensional indexing | `matrix[tuple_of_slices]` applies different slices per axis |

> **Key takeaway:** NumPy's multi-dimensional indexing with `slice` objects enables flexible, parametrized slicing across any number of axes.

---

### Task 101 — N-Dimensional Addition (`101-the_whole_barn.py`)

**Challenge:** Add two n-dimensional matrices recursively — generalizing element-wise addition to arbitrary depth.

**Approach:** Recursive function: base case is scalar addition (both are numbers). If elements are lists, recurse. Returns `None` on shape mismatch.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Recursive function for arbitrary nesting | Handle matrices of any depth with a single function |
| `isinstance(obj, list)` as recursion gate | Recurse if nested, add if scalar |
| Recursive error propagation | Return `None` up the call stack on shape mismatch |

> **Key takeaway:** Recursive addition handles arbitrary matrix dimensions by treating scalars as the base case — the same function works for 1D, 2D, 3D, and beyond.

---

### Task 102 — N-Dimensional Concatenation (`102-squashed_like_sardines.py`)

**Challenge:** Concatenate two n-dimensional matrices along a specified axis — the generalized version of Task 7.

**Approach:** Similar axis-based logic but designed for arbitrary depth. Axis=0 appends full sub-arrays; other axes perform nested concatenation.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Axis-aware conditional logic for n-D | Generalize 2D concatenation patterns to arbitrary dimensions |
| Sequential list operations for merging | `list.extend()` and `list.append()` for flexible joining |

> **Key takeaway:** Axis-aware concatenation generalizes 2D matrix joining to arbitrary dimensions — the axis parameter determines which level to merge at.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | Python list slicing `[start:stop]`, negative indexing | Python Slicing |
| 1 | Nested iteration + row-level slicing for 2D extraction | 2D Operations |
| 2 | `isinstance()` type checking, recursive descent for shape | Shape & Inspection |
| 3 | Nested loop coordinate swap for manual transpose | Manual Transpose |
| 4 | `len()` dimension validation, element-wise addition | 1D Operations |
| 5 | List comprehension for 2D pre-allocation | 2D Initialization |
| 6 | `list.extend()` for array concatenation | Concatenation |
| 7 | Axis parameter for 2D concatenation direction | Axis Control |
| 8 | Triple-nested loop dot product for matrix multiplication | Matrix Multiply |
| 9 | `np.array()`, `np.transpose()`, NumPy slicing | NumPy Basics |
| 10 | `array.shape` property (no-loop constraint) | NumPy Properties |
| 11 | `array.transpose()` method | NumPy Transpose |
| 12 | NumPy vectorized operators (`+`, `-`, `*`, `/`) | NumPy Broadcasting |
| 13 | `np.concatenate()` with axis parameter | NumPy Concatenation |
| 14 | `@` operator for matrix multiplication | NumPy Matmul |
| 100 | `slice()` objects, tuple indexing, `array.ndim` | Advanced Slicing |
| 101 | Recursive n-dimensional element-wise addition | Recursion |
| 102 | Axis-aware n-dimensional concatenation | Generalized Ops |

---

## Resources

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Array Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy `ndarray.shape` — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
- [NumPy `concatenate` — Official Docs](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
- [PEP 465 — `@` Matrix Multiplication Operator](https://peps.python.org/pep-0465/)
- [Python `isinstance()` — Official Docs](https://docs.python.org/3/library/functions.html#isinstance)