#!/usr/bin/env python3
"""Write a function def rename(df): that takes a
pd.DataFrame as input and performs the following:

df is a pd.DataFrame containing a column named Timestamp.
The function should rename the Timestamp column to Datetime.
Convert the timestamp values to datetime values
Display only the Datetime and Close column
Returns: the modified pd.DataFrame

"""

import pandas as pd


def rename(df):
    """takes a pd.DataFrame as input and performs ops

    Args:
        df (_type_): _description_
    """
    df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = df["Datetime"].Timestamp.to_pydatetime()
    df = df["Datetime", "Close"]
    return df
