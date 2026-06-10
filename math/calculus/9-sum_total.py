#!/usr/bin/env python3
"""Write a function def summation_i_squared(n)
    """


def summation_i_squared(n):
    """calculate sum

    Args:
        n (_type_): _description_
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
