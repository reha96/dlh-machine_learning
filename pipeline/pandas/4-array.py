#!/bin/usr/env python3
"""Write a function def array(df): that takes a
pd.DataFrame as input and performs the following:

df is a pd.DataFrame containing columns named
High and Close.
The function should select the last 10 rows of the
High and Close columns.
Convert these selected values into a numpy.ndarray.
Returns: the numpy.ndarray

"""
import pandas as pd


def array(df):
    df = df.tail(10)
    return df.to_numpy()
