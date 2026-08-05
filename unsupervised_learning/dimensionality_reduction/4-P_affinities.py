#!/usr/bin/env python3
"""Write a function that calculates the symmetric P affinities
    """
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
X is a numpy.ndarray of shape (n, d) containing the dataset
tol is the maximum tolerance allowed for the difference in Shannon entropy
from perplexity
perplexity is the perplexity that all Gaussian distributions should have

Returns:
P, a numpy.ndarray of shape (n, n) containing the symmetric P affinities
    """
    n, d = X.shape
    # step 1: get the distances, the empty P and the target entropy
    D, P, betas, H = P_init(X, perplexity)

    # step 2: find the right beta for each point with a binary search
    for i in range(n):
        low = None
        high = None
        beta = betas[i, 0]
        # distances from point i to every other point
        Di = np.delete(D[i], i)
        # entropy with the current beta
        Hi, Pi = HP(Di, beta)
        Hdiff = Hi - H
        # keep moving beta until the entropy matches H
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                # entropy too high: Gaussian too wide, so raise beta
                low = beta
                if high is None:
                    beta = beta * 2
                else:
                    beta = (beta + high) / 2
            else:
                # entropy too low: Gaussian too tight, so lower beta
                high = beta
                if low is None:
                    beta = beta / 2
                else:
                    beta = (beta + low) / 2
            Hi, Pi = HP(Di, beta)
            Hdiff = Hi - H
            tries += 1
        # step 3: store the beta and the affinities of point i
        betas[i, 0] = beta
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    # step 4: average the two halves and normalize
    P = (P + P.T) / (2 * n)

    return P
