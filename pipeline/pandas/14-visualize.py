#!/usr/bin/env python3
"""Complete the following script to visualize the pd.DataFrame:

The column Weighted_Price should be removed

Rename the column Timestamp to Date

Convert the timestamp values to date values

Index the data frame on Date

Missing values in Close should be set to the previous row value

Missing values in High, Low, Open should be set to the
same row's Close value

Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0

Plot the data from 2017 and beyond at daily intervals and
group the values of the same day such that:

High: max

Low: min

Open: mean

Close: mean

Volume(BTC): sum

Volume(Currency): sum

Return the transformed pd.DataFrame before plotting.

"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
