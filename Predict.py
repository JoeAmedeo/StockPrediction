import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

#Obtain data from CSVs
fundementalsData = pd.read_csv("./nyse/fundamentals.csv").dropna()
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
#display.display(finalDataFrame)

finalDataFrame = pd.get_dummies(finalDataFrame)
#display.display(finalDataFrame)

finalDataFrame = finalDataFrame.drop(columns=['initial', 'final'])
finalDataFrame = finalDataFrame.astype(np.float32)

scaler = MinMaxScaler(feature_range=(-1,1))

y_data = finalDataFrame['delta']
y_data[y_data < 0] = 0
y_data[y_data > 0] = 1
print((y_data == 0).sum())
print((y_data == 1).sum())
y_data = y_data.astype(np.int64)
x_data = finalDataFrame.drop(columns=['delta'])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=0)

grid = {
    'max_depth': range(1,11),
    'n_estimators': (10, 50, 100)
}

grid_svm = {
    'C': np.logspace(-3, 3, num=7, base=10),
    'gamma': np.logspace(-3, 3, num=7, base=10)
}


shuffle_split = ShuffleSplit(test_size=.8, train_size=.2, n_splits=5)

grid_search_random = GridSearchCV(RandomForestClassifier(), grid, cv=shuffle_split)
grid_search_gradient = GridSearchCV(GradientBoostingClassifier(), grid, cv=shuffle_split)
grid_search_svm = GridSearchCV(SVC(), grid_svm, cv=5)
grid_search_random.fit(x_train, y_train)
grid_search_gradient.fit(x_train, y_train)
grid_search_svm.fit(x_train, y_train)
print("Random Forest:")
print(grid_search_random.score(x_test, y_test))
print("Gradient Forest:")
print(grid_search_gradient.score(x_test, y_test))
print("SVM:")
print(grid_search_svm.score(x_test, y_test))