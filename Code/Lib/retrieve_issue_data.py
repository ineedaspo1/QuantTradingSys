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

def read_issue_data(issue, dataLoadStartDate, dataLoadEndDate):
    issue_name = issue + '.pkl'
    file_name = os.path.join(r'C:\Users\kruegkj\kevinkr OneDrive\OneDrive\IssueData\Equity', issue_name)
    
    #######################################
    # Download data from local file
    try:
        df = pd.read_pickle(file_name)
    except:
        print("No information for ticker '%s'" % issue)
    print (df.shape)
    print (df.head())
    
    print ("Successfully retrieved Primary")
    df = df.drop("Symbol", axis =1)
    df.set_index('Date', inplace=True)
    df['Pri'] = df.Close
    df2 = df.ix[dataLoadStartDate:dataLoadEndDate]
    return df2


if __name__ == "__main__":
    issue = "TLT"
    dataLoadStartDate = "2005-12-22"
    dataLoadEndDate = "2016-01-04"
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    print (dataSet.head(20))
