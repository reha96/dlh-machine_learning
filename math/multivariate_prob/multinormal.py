#!/usr/bin/env python3
"""Create the class MultiNormal that represents a
Multivariate Normal distribution:

class constructor def __init__(self, data):
data is a numpy.ndarray of shape (d, n) containing the data set:
n is the number of data points
d is the number of dimensions in each data point
If data is not a 2D numpy.ndarray, raise a TypeError with the message
data must be a 2D numpy.ndarray
If n is less than 2, raise a ValueError with the message
data must contain multiple data points
Set the public instance variables:
mean - a numpy.ndarray of shape (d, 1)
containing the mean of data
cov - a numpy.ndarray of shape (d, d)
containing the covariance matrix data

"""
import numpy as np


class MultiNormal:
    """represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """class constructor

        Args:
            data (_type_): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)  # shape (d, 1)
        inner_1 = data - self.mean
        self.cov = (inner_1 @ inner_1.T) / (n - 1)  # cov shape (d, d)

    def pdf(self, x):
        """public instance method that calculates the PDF
    x is a numpy.ndarray of shape (d, 1)
    containing the data point whose PDF should be calculated
    d is the number of dimensions of the Multinomial instance
    If x is not a numpy.ndarray, raise a TypeError with the message
    x must be a numpy.ndarray
    If x is not of shape (d, 1), raise a ValueError with the message
    x must have the shape ({d}, 1)
    Returns the value of the PDF
    You are not allowed to use the function numpy.cov

        Args:
            x (_type_): _description_

        Raises:
            TypeError: _description_
        """
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise TypeError("x must be a numpy.ndarray")
        d, n = x.shape
        if n != 1:
            raise ValueError(f"x must have the shape ({d}, 1)")
        left = (2*np.pi)**(-d/2) * np.linalg.det(self.cov)**(-1/2)
        mid = x - self.mean
        right = np.e**(-1/2 * mid.T @ np.linalg.inv(self.cov) @ mid)
        return left * right


if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal(
        [12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal(
        [12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
