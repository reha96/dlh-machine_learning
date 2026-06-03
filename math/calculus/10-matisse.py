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
    if not isinstance(poly, list):
        return None
    
    if not all(isinstance(item for item in poly, int)):
        return None
    out = []
    for i in range(1, len(poly)):
        out.append(poly[i]*i)
    return out
