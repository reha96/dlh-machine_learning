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

    if type(C) is float and C.is_integer():
        C = int(C)

    if not isinstance(poly, list) or len(poly) == 0 \
            or not isinstance(C, int):
        return None

    if not all(isinstance(i, (int, float)) for i in poly):
        return None

    out = [C]
    for i in range(len(poly)):
        out.append(poly[i]/(i+1))

    return out
