import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
import time
from datetime import datetime

#Obtain data from CSVs
fundementalsData = pd.read_csv("./nyse/fundamentals.csv")
pricesData = pd.read_csv("./nyse/prices.csv")

#Convert data to dataframes
fundementalsDataFrame = pd.DataFrame(fundementalsData)
pricesDataFrame = pd.DataFrame(pricesData)

#Manipulate data in fundementals csv

#Convert date into POSIX time
fundementalsDataFrame["Period Ending"] = pd.DatetimeIndex (fundementalsData["Period Ending"]).astype (np.int64)//(10**9)

#Convert column names for merge consistancy later
fundementalsDataFrame.columns = fundementalsDataFrame.columns.str.replace("Ticker Symbol","symbol")
fundementalsDataFrame.columns = fundementalsDataFrame.columns.str.replace("Period Ending","date")

#Manipulate price data

#Convert dates to POSIX time
pricesDataFrame["date"] = pd.DatetimeIndex (pricesDataFrame["date"]).astype (np.int64)//(10**9)

#Only need 3 columns
pricesDataFrame = pricesDataFrame[["date","symbol","open"]]
#Copy prices to new dataframe to get difference over time
priceDifferenceDataFrame = pricesDataFrame.copy()
#Subtract 60 days from copied data time, to get difference in price over 60 days
priceDifferenceDataFrame["date"] = priceDifferenceDataFrame["date"] - (86400 * 60)

#Merge original and coppied data and rename columns
mergeDataFrame = pd.merge(pricesDataFrame, priceDifferenceDataFrame, on=["symbol","date"])
mergeDataFrame.columns = ["date", "symbol", "initial", "final"]

#Add new column, being the difference in price over 60 days
mergeDataFrame["delta"] = mergeDataFrame["final"] - mergeDataFrame["initial"]

#Merge previous data with fundementals data
finalDataFrame = pd.merge(fundementalsDataFrame, mergeDataFrame, on=["symbol","date"])
