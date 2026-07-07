#!/usr/bin/env python3
"""Write a function def from_numpy(array): that creates
a pd.DataFrame from a np.ndarray:

array is the np.ndarray from which you should create the pd.DataFrame
The columns of the pd.DataFrame should be labeled in alphabetical order
and capitalized. There will not be more than 26 columns.
Returns: the newly created pd.DataFrame

"""
import pandas as pd


def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray:

    Args:
        array (_type_): _description_
    """

    df = pd.DataFrame(array)
    df.columns = [chr(65 + i) for i in range(len(df.columns))]  # list
    return df
