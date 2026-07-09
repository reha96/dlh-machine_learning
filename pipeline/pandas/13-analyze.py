#!/usr/bin/env python3
"""Write a function def analyze(df): that takes a pd.DataFrame and:

Computes descriptive statistics for all columns except the
Timestamp column.
Returns a new pd.DataFrame containing these statistics.

"""


def analyze(df):
    """takes a pd.DataFrame and do descriptive stats

    Args:
        df (_type_): _description_
    """
    _ = df.drop["Timestamp"]
    out = _.describe()
    return out
