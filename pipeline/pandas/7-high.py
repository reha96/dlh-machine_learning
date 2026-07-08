#!/usr/bin/env python3

"""Write a function def high(df): that takes a pd.DataFrame and:

Sorts it by the High price in descending order.
Returns: the sorted pd.DataFrame.
"""


def high(df):
    """takes a pd.DataFrame and:

    Sorts it by the High price in descending order.
    Returns: the sorted pd.DataFrame.


    Args:
        df (_type_): pd.DataFrame
    """

    df = df.sort_values("High", ascending=False)
    return df
