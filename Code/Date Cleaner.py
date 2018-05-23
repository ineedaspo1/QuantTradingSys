# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 08:18:37 2018

@author: kruegkj
"""

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import seaborn as sns
import random
from scipy import stats
import os

issue = 'AAPL'
issue_name = issue + '.pkl'
file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Equity', issue_name)
# Download data
try:
    #qt = pdr.DataReader(issue, data_source, start_date, end_date).reset_index()
    qt = pd.read_pickle(file_name)
except:
    print("No information for ticker '%s'" % issue)
print (qt.shape)
print (qt.head(40))

latest_date = qt["Date"].iloc[0]
earliest_date = qt["Date"].iloc[-1]
difference_in_years = relativedelta(earliest_date, latest_date).years

qt = qt.drop("Symbol", axis =1)
qt.set_index('Date', inplace=True)

qt['Close'].plot(figsize=(13,7))