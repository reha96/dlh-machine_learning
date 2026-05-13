#!/usr/bin/env python3
"""Write a function def np_elementwise(mat1, mat2): that performs element-wise addition, subtraction, multiplication, and division:

    You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
    You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
    You are not allowed to use any loops or conditional statements
    You can assume that mat1 and mat2 are never empty

    """


def np_elementwise(mat1, mat2):
    """function def np_elementwise(mat1, mat2): that performs element-wise addition, subtraction, multiplication, and
    division:

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
    """
    return (mat1.add(mat2), mat1.subtract(mat2),
            mat1.multiply(mat2), mat1.divide(mat2))
