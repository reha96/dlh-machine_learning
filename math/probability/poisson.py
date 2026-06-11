#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution:
Class contructor def __init__(self, data=None, lambtha=1.):
data is a list of the data to be used to estimate the distribution
lambtha is the expected number of occurences in a given time frame
Sets the instance attribute lambtha
Saves lambtha as a float
If data is not given, (i.e. None (be careful: not data
has not the same result as data is None)):
Use the given lambtha
If lambtha is not a positive value or equals to 0, raise
a ValueError with the message lambtha must be a positive value
If data is given:
Calculate the lambtha of data
If data is not a list, raise a TypeError with the message
data must be a list
If data does not contain at least two data points,
raise a ValueError with the message data must contain multiple values
"""


class Poisson:
    """class that represents a poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """class contructor

        Args:
            data (_type_, optional): _description_. Defaults to None.
            lambtha (_type_, optional): _description_. Defaults to 1..
        """
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is None:
            return None
        elif not isinstance(data, list):
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")

        sum = 0
        for i in range(len(data)):
            sum += data[i]
        self.lambtha = sum/len(data)
        return None

    def pmf(self, k):
        """Update the class Poisson:
        Instance method def pmf(self, k):
        Calculates the value of the PMF for
        a given number of "successes"
        k is the number of "successes"
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        Returns the PMF value for k
        Args:
            k (_type_): _description_
        """
        self.k = int(k)
        if not isinstance(self.k, int) or self.k < 0:
            return 0
        sigma = 1
        for i in range(1, self.k):
            sigma = sigma*i
        pmf = ((self.lambtha**self.k)*self.e **
               (-self.lambtha))/(self.k*sigma)
        return pmf

    def cdf(self, k):
        """Update the class Poisson:
        Instance method def cdf(self, k):
        Calculates the value of the CDF for a given number of "successes"
        k is the number of "successes"
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        Returns the CDF value for k
        Args:
            k (_type_): _description_
        """
        self.k = int(k)
        if not isinstance(self.k, int) or self.k < 0:
            return 0
        sigma = 1
        for j in range(1, self.k+1):
            sigma = sigma*j
            cdf = (self.lambtha**j)/sigma
        return cdf
