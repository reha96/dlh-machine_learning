#!/usr/bin/env python3
"""Create a class Binomial that represents a binomial distribution
"""


class Binomial:
    """class Binomial that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """class constructor

        Args:
            data (_type_, optional): _description_. Defaults to None.
            n (_type_, optional): _description_. Defaults to 1.
            p (_type_, optional): _description_. Defaults to 0.5.

        Raises:
            TypeError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.p = float(p)
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("n must be a positive value")
        if not (0 < self.p < 1):
            raise ValueError("p must be greater than 0 and less than 1")
        if data is None:
            return None
        elif not isinstance(data, list):
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")

        sum = 0
        for i in range(len(data)):
            sum += data[i]
        mean = sum / len(data)

        sigma = 0
        for i in range(len(data)):
            sigma += (data[i] - mean) ** 2
        variance = sigma / len(data)

        self.p = 1 - (variance / mean)
        self.n = round(mean / self.p)
        self.p = mean / self.n
        return None

    def pmf(self, k):
        """Instance method def pmf(self, k):
    Calculates the value of the PMF for a given number of "successes"
    k is the number of "successes"
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
    Returns the PMF value for k

        Args:
            k (_type_): _description_
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        a = 1.0
        for i in range(1, k + 1):
            a = a * (self.n - k + i) / i
        b = self.p ** k
        c = (1 - self.p) ** (self.n - k)
        pmf = a * b * c
        return pmf

    def cdf(self, k):
        """Instance method def cdf(self, k):
    Calculates the value of the CDF for a given number of "successes"
    k is the number of "successes"
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
    Returns the CDF value for k
    Hint: use the pmf method

        Args:
            k (_type_): _description_
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        cdf = 0.0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
