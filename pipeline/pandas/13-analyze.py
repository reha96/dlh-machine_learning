#!/usr/bin/env python3
"""Write a function def analyze(df): that takes a pd.DataFrame and:

Computes descriptive statistics for all columns except the
Timestamp column.
Returns a new pd.DataFrame containing these statistics.

"""
import pandas as pd


def analyze(df):
    """takes a pd.DataFrame and do descriptive stats

    Args:
        df (_type_): _description_
    """
    out = df.drop("Timestamp", axis=1)
    return out.describe()
