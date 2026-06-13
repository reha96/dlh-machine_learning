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
