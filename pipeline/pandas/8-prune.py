#!/usr/bin/env python3
"""Write a function def prune(df): that takes a pd.DataFrame and:

Removes any entries where Close has NaN values.
Returns: the modified pd.DataFrame.

"""


def prune(df):
    """Write a function def prune(df): that takes a pd.DataFrame and:

    Removes any entries where Close has NaN values.
    Returns: the modified pd.DataFrame.


    Args:
        df (_type_): _description_
    """

    df = df[df["Close"] != "NaN"]
    return df
