#!/usr/bin/env python3
"""Write a function def poly_derivative(poly): that calculates
    the derivative of a polynomial: 
    """


def poly_derivative(poly):
    """calculates
    the derivative of a polynomial

    Args:
        poly (_type_): _description_
    """
    if not isinstance(poly, list) and not all(isinstance(item, int)
                                              for item in poly):
        return None
    out = []
    for i in range(1, len(poly)):
        out.append(poly[i]*i)
    return out
