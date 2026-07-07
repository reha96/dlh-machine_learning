#!/usr/bin/env python3
"""Write a function def from_file(filename, delimiter):
that loads data from a file as a pd.DataFrame:

filename is the file to load from
delimiter is the column separator
Returns: the loaded pd.DataFrame

"""

import pandas as pd


def from_file(filename, delimiter):
    """loads data from a file as a pd.DataFrame

    Args:
        filename (_type_): _description_
        delimiter (_type_): _description_
    """

    df = pd.read_csv(filename, delimiter=delimiter)
    return df
