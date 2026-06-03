#!/usr/bin/env python3
"""Write a function def summation_i_squared(n)
    """


def summation_i_squared(n):
    """calculate sum

    Args:
        n (_type_): _description_
    """
    if not isinstance(n, int):
        return None

    if n == 1:
        return n

    return n**2 + summation_i_squared(n-1)
