import numpy as np

# Helper functions


def minor_matrix(A):
    """Compute the matrix of minors for a square matrix."""
    n = A.shape[0]
    M = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(n):
            # Delete row i, column j
            sub = np.delete(np.delete(A, i, axis=0), j, axis=1)
            M[i, j] = np.linalg.det(sub)
    return M


def cofactor_matrix(A):
    """Compute the cofactor matrix from the matrix of minors."""
    M = minor_matrix(A)
    n = A.shape[0]
    signs = (-1) ** (np.arange(n)[:, None] + np.arange(n)[None, :])
    return signs * M


def adjugate_matrix(A):
    """Adjugate = transpose of cofactor matrix."""
    return cofactor_matrix(A).T


def classify_definiteness(A, tol=1e-10):
    eigvals = np.linalg.eigvalsh(A)   # works for symmetric matrices
    if np.all(eigvals > tol):
        return "Positive definite"
    elif np.all(eigvals >= -tol):     # allows near-zero eigenvalues
        return "Positive semi-definite"
    elif np.all(eigvals < -tol):
        return "Negative definite"
    elif np.all(eigvals <= tol):
        return "Negative semi-definite"
    else:
        return "Indefinite"


# ========== Question 0: Determinant ==========
A0 = np.array([[-7, 0, 6],
               [5, -2, -10],
               [4, 3, 2]])
det0 = np.linalg.det(A0)
print(f"Q0: Determinant = {det0:.0f}  -> should be -44")

# ========== Question 1: Minor ==========
A1 = A0.copy()
M1 = minor_matrix(A1)
print("\nQ1: Minor matrix:")
print(M1)
options_q1 = [
    np.array([[26, 50, 23], [-18, -38, -21], [12, 40, 15]]),
    np.array([[26, 50, 23], [-18, -38, -21], [12, 40, 14]]),
    np.array([[26, 50, 23], [-18, -39, -21], [12, 40, 14]]),
    np.array([[26, 50, 23], [-18, -39, -21], [12, 40, 15]])
]
for i, opt in enumerate(options_q1, 1):
    if np.allclose(M1, opt):
        print(f"Matches option {i}")

# ========== Question 2: Cofactor ==========
A2 = np.array([[6, -9, 9],
               [7, 5, 0],
               [4, 3, -8]])
C2 = cofactor_matrix(A2)
print("\nQ2: Cofactor matrix:")
print(C2)
options_q2 = [
    np.array([[-40, 56, 1], [-45, -84, -54], [-45, 64, 93]]),
    np.array([[-40, 56, 1], [-44, -84, -54], [-45, 64, 93]]),
    np.array([[-40, 56, 1], [-44, -84, -54], [-45, 63, 93]]),
    np.array([[-40, 56, 1], [-45, -84, -54], [-45, 63, 93]])
]
for i, opt in enumerate(options_q2, 1):
    if np.allclose(C2, opt):
        print(f"Matches option {i}")

# ========== Question 3: Adjugate ==========
A3 = np.array([[-4, 1, 9],
               [-9, -8, -5],
               [-3, 8, 10]])
adj3 = adjugate_matrix(A3)
print("\nQ3: Adjugate matrix:")
print(adj3)
options_q3 = [
    np.array([[-40, 62, 67], [105, -13, -101], [-97, 29, 41]]),
    np.array([[-40, 62, 67], [105, -14, -101], [-97, 29, 41]]),
    np.array([[-40, 62, 67], [105, -13, -101], [-96, 29, 41]]),
    np.array([[-40, 62, 67], [105, -14, -101], [-96, 29, 41]])
]
for i, opt in enumerate(options_q3, 1):
    if np.allclose(adj3, opt):
        print(f"Matches option {i}")

# ========== Question 4: Inverse ==========
A4 = np.array([[1, 0, 1],
               [2, 1, 2],
               [1, 0, -1]])
det4 = np.linalg.det(A4)
if np.isclose(det4, 0):
    print("\nQ4: Matrix is singular")
else:
    inv4 = np.linalg.inv(A4)
    print("\nQ4: Inverse matrix:")
    print(inv4)
    options_q4 = {
        "1": np.array([[0.5, 0, 0.5], [0, 1, 2], [0.5, 0, 0.5]]),
        "2": np.array([[0.5, 0, 0.5], [-2, 1, 0], [0.5, 0, -0.5]]),
        "3": np.array([[0.5, 0, 0.5], [2, 1, 0], [0.5, 0, 0.5]])
    }
    for key, opt in options_q4.items():
        if np.allclose(inv4, opt):
            print(f"Matches option {key}")

# ========== Question 5: Inverse or singular ==========
A5 = np.array([[2, 1, 2],
               [1, 0, 1],
               [4, 1, 4]])
det5 = np.linalg.det(A5)
if np.isclose(det5, 0):
    print("\nQ5: Matrix is singular (det = 0)")
else:
    inv5 = np.linalg.inv(A5)
    print("\nQ5: Inverse matrix:", inv5, sep="\n")

# ========== Question 6: A^10 v ==========
A6 = np.array([[-2, -4, 2],
               [-2, 1, 2],
               [4, 2, 5]])
v6 = np.array([2, -3, -1])
# Check that v is an eigenvector and find lambda
Av = A6 @ v6
lambda6 = Av[0] / v6[0]   # safe because v6[0] != 0
A10v = (lambda6 ** 10) * v6
print("\nQ6: A^10 v =", A10v)
options_q6 = [
    np.array([118098, -177147, -59049]),
    np.array([2097152, -3145728, -1048576]),
    np.array([2048, -3072, -1024])
]
for i, opt in enumerate(options_q6, 1):
    if np.allclose(A10v, opt):
        print(f"Matches option {i}")

# ========== Question 7: Other eigenvalues/vectors ==========
print("\nQ7: Checking candidate eigenvalue/vector pairs:")
eigvals7, eigvecs7 = np.linalg.eig(A6)
# Candidates
candidates = [
    (5, [2, 1, 1]),
    (-5, [-2, -1, 1]),
    (-3, [4, -2, 3]),
    (6, [1, 6, 16])
]
for lam, v in candidates:
    res = A6 @ v - lam * np.array(v)
    if np.allclose(res, 0):
        print(
            f"λ = {lam}, v = {v} -> IS an eigenpair (matches option {candidates.index((lam, v))+1})")

# ========== Question 8: Definiteness ==========
A8 = np.array([[-1, 2, 0],
               [2, -5, 2],
               [0, 2, -6]])
print("\nQ8: Definiteness =", classify_definiteness(A8))

# ========== Question 9: Definiteness ==========
A9 = np.array([[2, 2, 1],
               [2, 1, 3],
               [1, 3, 8]])
print("Q9: Definiteness =", classify_definiteness(A9))

# ========== Question 10: Definiteness ==========
A10 = np.array([[2, 1, 1],
                [1, 2, -1],
                [1, -1, 2]])
print("Q10: Definiteness =", classify_definiteness(A10))
