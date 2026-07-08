#!/usr/bin/env python3
"""Write a function def flip_switch(df): 
that takes a pd.DataFrame and:

Sorts the data in reverse chronological order.
Transposes the sorted dataframe.
Returns: the transformed pd.DataFrame.

"""


def flip_switch(df):
    """takes a pd.DataFrame and ops

    Args:
        df (_type_): pd.DataFrame
    """

    df = df.sort_values("Timestamp", ascending=False)
    df = df.T
    return df
