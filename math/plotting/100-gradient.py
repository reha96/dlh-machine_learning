#!/usr/bin/env python3
"""Complete the following source code to create
a scatter plot of sampled elevations on a mountain:

The x-axis should be labeled x coordinate (m)
The y-axis should be labeled y coordinate (m)
The title should be Mountain Elevation
A colorbar should be used to display elevation
The colorbar should be labeled elevation (m)

"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """scatter with gradient colors

    Returns:
        _type_: _description_
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.title("Mountain Elevation")

    plt.scatter(x, y, c=z)
    plt.colorbar(label='elevation (m)')
    return plt.show()
