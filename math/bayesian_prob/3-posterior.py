#!/usr/bin/env python3
"""Based on 2-marginal.py, write a function
def posterior(x, n, P, Pr):
that calculates the posterior probability
for the various hypothetical
probabilities of developing severe
side effects given the data:

x is the number of patients that develop severe side effects
n is the total number of patients observed
P is a 1D numpy.ndarray containing the various hypothetical
probabilities of developing severe side effects
Pr is a 1D numpy.ndarray containing the prior beliefs of P
If n is not a positive integer, raise a ValueError with the
message n must be a positive integer
If x is not an integer that is greater than or equal to 0,
raise a ValueError with the message x must be an integer that
is greater than or equal to 0
If x is greater than n, raise a ValueError with the message
x cannot be greater than n
If P is not a 1D numpy.ndarray, raise a TypeError with the message
P must be a 1D numpy.ndarray
If Pr is not a numpy.ndarray with the same shape as P, raise
a TypeError with the message Pr must be a numpy.ndarray with
the same shape as P
If any value in P or Pr is not in the range
[0, 1], raise a ValueError
with the message All values in {P} must be in the
range [0, 1] where
{P} is the incorrect variable
If Pr does not sum to 1, raise a ValueError with the message Pr
must sum to 1
All exceptions should be raised in the above order
Returns: the posterior probability of each probability
in P given x and n, respectively

"""
import numpy as np


def likelihood(x, n, P):
    """calculates the
likelihood of obtaining data

    Args:
        x (_type_): _description_
        n (_type_): _description_
        P (_type_): _description_
    """
    m1 = "x must be an integer that is greater than or equal to 0"
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(m1)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    m2 = "All values in P must be in the range [0, 1]"
    for i in range(len(P)):
        if not (0 <= P[i] <= 1):
            raise ValueError(m2)

    a = 1.0
    for i in range(1, x + 1):
        a = a * (n - x + i) / i
    b = P ** x
    c = (1 - P) ** (n - x)
    likelihood = a * b * c
    return likelihood


def intersection(x, n, P, Pr):
    """calculates the intersection of obtaining this
    data with the various hypothetical probabilities

    Args:
        x (_type_): _description_
        n (_type_): _description_
        P (_type_): _description_
        Pr (_type_): _description_
    """

    m1 = "x must be an integer that is greater than or equal to 0"
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(m1)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    m4 = "Pr must be a numpy.ndarray with the same shape as P"
    if not isinstance(Pr, np.ndarray) or np.shape(Pr) != np.shape(P):
        raise TypeError(m4)

    m2 = "All values in P must be in the range [0, 1]"
    m3 = "All values in Pr must be in the range [0, 1]"
    for i in range(len(P)):
        if not (0 <= P[i] <= 1):
            raise ValueError(m2)
    for i in range(len(Pr)):
        if not (0.0 <= Pr[i] <= 1.0):
            raise ValueError(m3)
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    a = 1.0
    for i in range(1, x + 1):
        a = a * (n - x + i) / i
    b = P ** x
    c = (1 - P) ** (n - x)
    likelihood = a * b * c
    return Pr*likelihood


def marginal(x, n, P, Pr):
    """calculates the marginal probability of obtaining the data

    Args:
        x (_type_): _description_
        n (_type_): _description_
        P (_type_): _description_
        Pr (_type_): _description_
    """
    m1 = "x must be an integer that is greater than or equal to 0"
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(m1)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    m4 = "Pr must be a numpy.ndarray with the same shape as P"
    if not isinstance(Pr, np.ndarray) or np.shape(Pr) != np.shape(P):
        raise TypeError(m4)

    m2 = "All values in P must be in the range [0, 1]"
    m3 = "All values in Pr must be in the range [0, 1]"
    for i in range(len(P)):
        if not (0 <= P[i] <= 1):
            raise ValueError(m2)
    for i in range(len(Pr)):
        if not (0.0 <= Pr[i] <= 1.0):
            raise ValueError(m3)
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    return np.sum(likelihood(x, n, P) * Pr)


def posterior(x, n, P, Pr):
    """calculates the posterior probability
for the various hypothetical
probabilities

    Args:
        x (_type_): _description_
        n (_type_): _description_
        P (_type_): _description_
        Pr (_type_): _description_
    """
    m1 = "x must be an integer that is greater than or equal to 0"
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(m1)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    m4 = "Pr must be a numpy.ndarray with the same shape as P"
    if not isinstance(Pr, np.ndarray) or np.shape(Pr) != np.shape(P):
        raise TypeError(m4)

    m2 = "All values in P must be in the range [0, 1]"
    m3 = "All values in Pr must be in the range [0, 1]"
    for i in range(len(P)):
        if not (0 <= P[i] <= 1):
            raise ValueError(m2)
    for i in range(len(Pr)):
        if not (0.0 <= Pr[i] <= 1.0):
            raise ValueError(m3)
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
