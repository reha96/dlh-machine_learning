#!/usr/bin/env python3
"""Write a function def poly_integral(poly, C=0):
    that calculates the integral of a polynomial:
    """


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial

    Args:
        poly (_type_): _description_
        C (int, optional): _description_. Defaults to 0.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    out = [0]
    for i in range(len(poly)):
        out.append(poly[i]/(i+1))

    if not isinstance(C, float) or all(x == 0 for x in out):
        return None

    return out
