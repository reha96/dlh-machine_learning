#!/usr/bin/env python3
"""Write a function that performs K-means on a dataset:
    """


import sklearn.cluster


def kmeans(X, k):
    """
X is a numpy.ndarray of shape (n, d) containing the dataset
k is the number of clusters

Returns: C, clss

C is a numpy.ndarray of shape (k, d) containing the centroid means

clss is a numpy.ndarray of shape (n,) containing the index of the
cluster in C that each data point belongs to
    """
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
