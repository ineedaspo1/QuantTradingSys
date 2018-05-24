# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:25:39 2018

@author: kruegkj

sample_model.py

First pass on modeling of input dataframe

Inputs:
    Dataframe
    Timeframe?
    Model names
    Variables to drop?
Outputs:
    Model results
    Saved model?
    
From main, create many of the vairables created as defaults in other modules

"""
import sys
sys.path.append('../lib')
sys.path.append('../transform')
sys.path.append('../indicators')
sys.path.append('../predictors')
sys.path.append('../utilities')



def main(): 
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    print(issue)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    return dataSet

if __name__ == "__main__":
    issue = "xly"
    lookback = 16
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30" 
    dataSet = main()
    
    beLongThreshold = 0
    
    targetDataSet = setTarget(dataSet, "Long", beLongThreshold)
    nrows = targetDataSet.shape[0]
    print ("nrows: ", nrows)
    print (targetDataSet.shape)
    print (targetDataSet.tail(10))
    
    print ("beLong counts: ")
    print (targetDataSet['beLong'].value_counts())
    print ("out of ", nrows)
    
    