# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:17:54 2018

@author: kruegkj

retrieve_issue_data.py
"""

from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn import mixture as mix
import seaborn as sns 
import matplotlib.pyplot as plt
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def read_issue_data(issue, dataLoadStartDate, dataLoadEndDate):
    us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    issue_name = issue + '.pkl'
    file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Equity', issue_name)
    
    #######################################
    # Download data from local file
    try:
        df = pd.read_pickle(file_name)
    except:
        print("================================")
        print("No information for ticker '%s'" % issue)
        print("================================")
        raise SystemExit
    #print (df.shape)
    #print (df.head(20))
    
    print ("Successfully retrieved Primary")
    df = df.drop("Symbol", axis =1)
    df.set_index('Date', inplace=True)
    df['Pri'] = df.Close
    df2 = pd.date_range(start=dataLoadStartDate, end=dataLoadEndDate, freq=us_cal)
    df3 = df.reindex(df2)
    return df3


if __name__ == "__main__":
    issue = "TLT"
    dataLoadStartDate = "2017-04-01"
    dataLoadEndDate = "2018-04-01"
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    nrows = dataSet.shape[0]
    print (dataSet.shape)
    print (dataSet.head(20))
    
    plt.style.use('seaborn-ticks')
    fig, ax = plt.subplots()
    
    plt.plot(dataSet['Close'], label=issue)
    plt.legend(loc='upper left')
    for ax in axes:
            ax.label_outer()
            ax.legend(loc='upper left', frameon=True, fontsize=8)
            ax.grid(True, which='both')
            fig.autofmt_xdate()
            ax.xaxis_date()
            ax.autoscale_view()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            ax.minorticks_on()
            ax.tick_params(axis='y',which='minor',bottom='off')
    
    