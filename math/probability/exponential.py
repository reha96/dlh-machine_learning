#!/usr/bin/env python3
"""Create a class Exponential that represents an exponential
distribution:

Class contructor def __init__(self, data=None, lambtha=1.):
data is a list of the data to be used to estimate the distribution
lambtha is the expected number of occurences in a given time frame
Sets the instance attribute lambtha
Saves lambtha as a float
If data is not given (i.e. None):
Use the given lambtha
If lambtha is not a positive value, raise a ValueError
with the message lambtha must be a positive value
If data is given:
Calculate the lambtha of data
If data is not a list, raise a TypeError with the message data must
be a list
If data does not contain at least two data points, raise a
ValueError with the message data must contain multiple values
    """


class Exponential:
    """class that represents an exponential
distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """class constructor

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
        self.lambtha = 1/(sum/len(data))
        return None

    def pdf(self, x):
        """Update the class Exponential:
        Instance method def pdf(self, x):
        Calculates the value of the PDF for a given time period
        x is the time period
        Returns the PDF value for x
        If x is out of range, return 0
        Args:
            x (_type_): _description_
        """
        x = float(x)
        if not isinstance(x, float) or x < 0:
            return 0
        pmf = self.lambtha*self.e**(-self.lambtha*x)
        return pmf
