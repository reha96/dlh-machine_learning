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

df = df.drop("Weighted_Price", axis=1)  # drop col

df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')  # convert to date

df = df.set_index("Date")  # set index

df['Close'] = df['Close'].ffill()  # by default previous row

for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

df[['Volume_(BTC)', 'Volume_(Currency)']] = \
    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0) # missing vals
    
df = df[df.index >= '2017-01-01'] # keep data after 1st Jan 2017

# resample to daily frequency and aggregate
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# plot example: show Close and Volume
df[['Close', 'Volume_(BTC)']].plot(subplots=True, figsize=(10, 6))
plt.show()


