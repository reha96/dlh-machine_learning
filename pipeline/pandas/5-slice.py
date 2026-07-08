#!/usr/bin/env python3
"""Write a function def slice(df): that takes a pd.DataFrame and:

Extracts the columns High, Low, Close, and Volume_(BTC).
Selects every 60th row from these columns.
Returns: the sliced pd.DataFrame

"""


def slice(df):
    """takes a pd.DataFrame and ops

    Args:
        df (_type_): _description_
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    df = df[::60]
    return df
