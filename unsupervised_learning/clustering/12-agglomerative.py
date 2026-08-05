#!/usr/bin/env python3
"""Write a function that performs agglomerative clustering on a dataset:
    """

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """    
    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color

    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
    indices for each data point
    """
    # Perform hierarchical/agglomerative clustering, Z is linkage mat
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    # Form flat clusters from the hierarchical clustering defined by Z
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')
    # plot
    scipy.cluster.hierarchy.dendrogram(
        Z, color_threshold=dist, above_threshold_color='grey')
    plt.show()
    return clss
