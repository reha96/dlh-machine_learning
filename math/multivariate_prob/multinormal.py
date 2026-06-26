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
You are not allowed to use the function numpy.cov

"""
import numpy as np


class MultiNormal:
    """represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        n, d = data.shape
        print(f"n = {n}, d = {d}")
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=0, keepdims=True).T  # shape (d, 1)
        inner_1 = data - self.mean.T  # transpose so operation feasible
        # sample cov with shape (d, d)
        self.cov = (inner_1 @ inner_1.T) / (n - 1)


if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal(
        [12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    print(f"mean dim: {mn.mean.shape}, cov din: {mn.cov.shape}")
    print(mn.mean)
    print(mn.cov)
