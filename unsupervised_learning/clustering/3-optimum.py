#!/usr/bin/env python3
"""Write a function that tests for the optimum number of clusters
by variance:
    """


import numpy as np
import matplotlib.pyplot as plt


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """X is a numpy.ndarray of shape (n, d) containing the data set

kmin is a positive integer containing the minimum number of clusters
to check for (inclusive)

kmax is a positive integer containing the maximum number of clusters
to check for (inclusive)

iterations is a positive integer containing the maximum number of
iterations for K-means

This function should analyze at least 2 different cluster sizes

You may use at most 2 loops

Returns: results, d_vars, or None, None on failure
results is a list containing the outputs of K-means for each
cluster size
d_vars is a list containing the difference in variance from
the smallest cluster size for each cluster size

    """
    if not isinstance(kmin, int) or kmin <= 0 \
        or not isinstance(kmax, int) or kmax <= 0 \
            or not isinstance(iterations, int) or iterations <= 0:
        return (None, None)
    else:
        try:
            kmeans = __import__('1-kmeans').kmeans
            variance = __import__('2-variance').variance
            
            d_vars = 0
            for i in range(kmin, kmax+1):
                results = kmeans(X, i, iterations)
                var = variance(X,results[0])
                d_vars += var
            
            return (results, d_vars)
        except Exception:
            return (None, None)


if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    print(results)
    print(np.round(d_vars, 5))
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()
