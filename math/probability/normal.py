#!/usr/bin/env python3
"""Create a class Normal that represents a normal distribution:

Class contructor def __init__(self, data=None, mean=0., stddev=1.):
data is a list of the data to be used to estimate the distribution
mean is the mean of the distribution
stddev is the standard deviation of the distribution
Sets the instance attributes mean and stddev
Saves mean and stddev as floats
If data is not given (i.e. None (be careful: not data has not
the same result as data is None))
Use the given mean and stddev
If stddev is not a positive value or equals to 0, raise a ValueError
with the message stddev must be a positive value
If data is given:
Calculate the mean and standard deviation of data
If data is not a list, raise a TypeError with the message
data must be a list
If data does not contain at least two data points, raise a ValueError
with the message data must contain multiple values

"""


class Normal:
    """class Normal that represents a normal distribution
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """class constructor

        Args:
            data (_type_, optional): _description_. Defaults to None.
            mean (_type_, optional): _description_. Defaults to 0..
            stddev (_type_, optional): _description_. Defaults to 1..

        Raises:
            TypeError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.stddev = float(stddev)
        self.mean = float(mean)
        if self.stddev <= 0:
            raise ValueError("stddev must be a positive value")
        if data is None:
            return None
        elif not isinstance(data, list):
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")

        sum = 0
        for i in range(len(data)):
            sum += data[i]
        self.mean = sum/len(data)

        sigma = 0
        for i in range(len(data)):
            sigma += (data[i]-self.mean)**2
        self.stddev = (sigma/len(data))**(1/2)
        return None

    def z_score(self, x):
        """Update the class Normal:
    Instance method def z_score(self, x):
        Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x"""
        self.x = x
        z = (x - self.mean)/self.stddev
        return z

    def x_value(self, z):
        """
    Instance method def x_value(self, z):
        Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        Args:
            x (_type_): _description_
        """
        self.z = z
        x = self.stddev*self.z + self.mean
        return x

    def pdf(self, x):
        """Instance method def pdf(self, x):
    Calculates the value of the PDF for a given x-value
    x is the x-value
    Returns the PDF value for x

        Args:
            x (_type_): _description_
        """
        self.x = x
        pdf = (1/((2*self.pi*self.stddev**2)**1/2))*self.e **\
            ((-(self.x-self.mean)**2)/(2*self.stddev**2))
        return pdf
