# Math — Calculus

A progressive journey through single-variable and multivariable calculus — from summation and product notation to derivatives, integrals, and Python implementations of polynomial differentiation and integration.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Evaluate summations using $\Sigma$ (sigma) notation |
| 2 | Evaluate products using $\Pi$ (pi) notation |
| 3 | Compute derivatives of polynomial, exponential, and logarithmic functions |
| 4 | Compute partial derivatives of multivariable functions |
| 5 | Evaluate indefinite and definite integrals |
| 6 | Evaluate double integrals over rectangular regions |
| 7 | Implement polynomial derivative calculation in Python |
| 8 | Implement polynomial integral (antiderivative) calculation in Python |
| 9 | Apply the closed-form formula for $\sum_{i=1}^{n} i^2$ |

---

## Module Structure

This module contains two types of tasks:

| Type | Files | Description |
|------|-------|-------------|
| **Computation Exercises** | 0–8, 11–16 | Hand-calculated answers stored as single-number files |
| **Python Implementations** | 9, 10, 17 | Functions that compute sums, derivatives, and integrals programmatically |

---

## Computation Exercises (Tasks 0–8, 11–16)

These tasks build calculus intuition through manual computation. Each file stores the numeric answer.

| Task | File | Topic | Concept Tested |
|------|------|-------|----------------|
| 0 | `0-sigma_is_for_sum` | Sigma notation | $\sum_{i=2}^{5} i$ |
| 1 | `1-seegma` | Sigma notation | $\sum_{k=1}^{4} k^2$ |
| 2 | `2-pi_is_for_product` | Pi notation | $\prod_{i=1}^{4} i$ |
| 3 | `3-pee` | Pi notation | Product with expressions |
| 4 | `4-hello_derivatives` | Derivatives | $f'(x)$ for polynomial $f$ |
| 5 | `5-log_on_fire` | Derivatives | Derivative of $\ln(x)$ and $e^x$ |
| 6 | `6-voltaire` | Derivatives | Derivative at a specific point |
| 7 | `7-partial_truths` | Partial derivatives | $\frac{\partial f}{\partial x}$ |
| 8 | `8-all-together` | Mixed derivatives | Combined partial derivative evaluation |
| 11 | `11-integral` | Indefinite integrals | $\int x^n \, dx$ |
| 12 | `12-integral` | Indefinite integrals | $\int e^x \, dx$ |
| 13 | `13-definite` | Definite integrals | $\int_a^b f(x) \, dx$ |
| 14 | `14-definite` | Definite integrals | Area under a curve |
| 15 | `15-definite` | Definite integrals | Definite integral with substitution |
| 16 | `16-double` | Double integrals | $\iint f(x,y) \, dx \, dy$ |

---

## Task-by-Task Reference (Python Implementations)

Each task below highlights the **unique challenge** it posed and the **new technique** introduced.

---

### Task 9 — Summation Formula (`9-sum_total.py`)

**Challenge:** Compute $\sum_{i=1}^{n} i^2$ efficiently without a loop — introducing the closed-form mathematical formula.

**Approach:** Implement the formula $\frac{n(n+1)(2n+1)}{6}$ using integer arithmetic. Validate that $n$ is a positive integer. Use `//` for floor division since the numerator is always divisible by 6.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| $\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6}$ | Closed-form sum of squares — O(1) instead of O(n) |
| `//` integer floor division | Guarantee integer result when numerator is divisible by denominator |
| Input validation: `isinstance(n, int) and n >= 1` | Reject non-integer and non-positive inputs |
| `return None` for invalid input | Sentinel pattern for invalid parameters |

> **Key takeaway:** Mathematical formulas can replace loops. The sum of squares formula is a classic closed form — knowing it turns an O(n) problem into O(1).

---

### Task 10 — Polynomial Derivative (`10-matisse.py`)

**Challenge:** Compute the derivative of a polynomial represented as a list of coefficients — implementing the power rule programmatically.

**Approach:** Given coefficients $[a_0, a_1, a_2, \dots]$ representing $a_0 + a_1x + a_2x^2 + \dots$, the derivative is $a_1 + 2a_2x + 3a_3x^2 + \dots$. For each coefficient at index $i$, multiply by $i$ (the power) and shift left by 1 position. Handle edge cases: empty list → `None`, constant polynomial → `[0]`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Power rule: $\frac{d}{dx}(a_i x^i) = i \cdot a_i x^{i-1}$ | Derivative of each term is coefficient × power |
| Coefficient list representation | Index = power, value = coefficient |
| `poly[i] * i` for derivative coefficient | Multiply by the exponent (index) |
| `all(x == 0 for x in out)` → return `[0]` | Handle the zero-polynomial edge case |

> **Key takeaway:** The derivative of $a_n x^n$ is $n \cdot a_n x^{n-1}$. In a coefficient list, you multiply each coefficient by its index (the power) and shift left. The constant term (index 0) is dropped.

---

### Task 17 — Polynomial Integral (`17-integrate.py`)

**Challenge:** Compute the indefinite integral (antiderivative) of a polynomial — implementing the reverse power rule with an integration constant.

**Approach:** For each coefficient $a_i$ at index $i$ (representing $a_i x^i$), the integral term is $\frac{a_i}{i+1} x^{i+1}$. Store at index $i+1$ in the output. The constant $C$ goes at index 0. Use integer simplification: if the division is exact, store as `int`; otherwise store as `float`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Reverse power rule: $\int a_i x^i \, dx = \frac{a_i}{i+1} x^{i+1} + C$ | Integral of each term divides by the new exponent |
| `C` as the integration constant | Arbitrary constant added to every indefinite integral |
| Index shift: input[i] → output[i+1] | The integral increases each term's degree by 1 |
| `int(val)` if exact else `float(val)` | Preserve exact integer results, use float for fractions |
| `C.is_integer()` float check | Handle the case where C is a float that represents a whole number |

> **Key takeaway:** Integration is the inverse of differentiation. Each term $a x^n$ becomes $\frac{a}{n+1} x^{n+1}$. The integration constant $C$ represents the family of all antiderivatives. Don't forget to shift indices — integrating increases each power by 1.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0–8 | Manual computation: $\Sigma$, $\Pi$, derivatives, partial derivatives | Manual Calculus |
| 9 | Closed-form $\sum i^2$ formula, `//` integer division | Summation |
| 10 | Polynomial derivative via power rule, coefficient list representation | Derivatives |
| 11–16 | Manual computation: indefinite, definite, double integrals | Manual Calculus |
| 17 | Polynomial integral via reverse power rule, integration constant $C$ | Integration |

---

## Resources

- [Summation Notation — Wikipedia](https://en.wikipedia.org/wiki/Summation)
- [Power Rule — Wikipedia](https://en.wikipedia.org/wiki/Power_rule)
- [Fundamental Theorem of Calculus — Wikipedia](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus)
- [Partial Derivative — Wikipedia](https://en.wikipedia.org/wiki/Partial_derivative)
- [Double Integral — Wikipedia](https://en.wikipedia.org/wiki/Multiple_integral)