# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:08:31 2018

@author: kruegkj

plot_functions.py


""" 
from datetime import datetime
import numpy as np
import retrieve_issue_data
import matplotlib.pyplot as plt

def set_plot(startDate, endDate, dataFrame):
    testFirstYear = startDate
    testFinalYear = endDate
    qtPlot = dataFrame.ix[testFirstYear:testFinalYear]
    plt.figure(1)    
    qtPlot['Close'].plot(figsize=(13,4  ))
#    plt.figure(2)
#    qtPlot['beLong'].plot(figsize=(13,4  ))
    
if __name__ == "__main__":
    issue = "XRT"
    dataLoadStartDate = "1998-12-22"
    dataLoadEndDate = "2016-01-04"
    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    nrows = dataSet.shape[0]
    print ("nrows: ", nrows)
    startDate = "2015"
    endDate = "2015"
    set_plot(startDate, endDate, dataSet)