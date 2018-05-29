# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:39:48 2018

@author: kruegkj

signal_stationarity_tests.py
"""

import sys
sys.path.append('../lib')

# Import the Time Series library
import statsmodels.tsa.stattools as ts
from retrieve_issue_data import read_issue_data
from stat_tests import *
from compute_target import *
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 14,3

"""
Add modules
1. Inputs: issue, years to load, segments, endDate, beLong threshold
2. 
"""
def set_segment_and_load_signal(issue, years_to_load, segments, endDate, beLongThreshold):
    segment_months = int((years_to_load*12)/segments)
    dataLoadEndDate = endDate
    dataLoadStartDate = dataLoadEndDate - relativedelta(years=years_to_load)
    print(dataLoadStartDate)
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    targetDataSet = setTarget(dataSet, "Long", beLongThreshold)
    nrows = targetDataSet.shape[0]
    print ("beLong counts: ")
    print (targetDataSet['beLong'].value_counts())
    print ("out of ", nrows)
    
        
    # increment through time segments
    i = 0
    startPlotDate = dataLoadStartDate
    while i < segments:
        endPlotDate = startPlotDate + relativedelta(months=segment_months)
        qtPlot = targetDataSet.ix[startPlotDate:endPlotDate]   
        """
        # Plot price and belong indicator
        fig = plt.figure(figsize=(15,8  ))
        ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                           xticklabels=[])
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                           ylim=(-1,1))
        x = np.linspace(0, 10)
        ax1.plot(qtPlot['Close'])
        ax2.plot(qtPlot['beLong'],color='red');
    
        orig = plt.plot(qtPlot.Close, color='blue', label=issue)
        plt.legend(loc='best')
        print("\n\n\n")
        plt.show(block=False)
        """
        numSubPlots = 2
        # format the ticks
        fig, axes = plt.subplots(numSubPlots,1, figsize=(numSubPlots*4,6), sharex=True)
        
        axes[0].plot(qtPlot['Close'], label=issue)
        axes[1].plot(qtPlot['beLong'], color='red', label='beLong');
        plt.legend(loc='best')
        # Bring subplots close to each other.
        plt.subplots_adjust(hspace=0.1)
        
        #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
        # Hide x labels and tick labels for all but bottom plot.
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
        print("\n\n\n")
        plt.show(block=False)

        startPlotDate = endPlotDate + relativedelta(days=1)
        i+=1
      
        adf_test_signal(qtPlot,issue)
        hurst_setup(qtPlot['beLong'][:],issue)
        
        dBeLong = qtPlot.beLong.values
        mean_and_variance(dBeLong)
 
    
if __name__ == "__main__":
    issue = "tlt"
    years_to_load = 1
    segments = 6
    dataLoadEndDate = datetime.date(2018, 4, 1)
    beLongThreshold = 0
    #segment_months = int((years_to_load*12)/segments)
    
    # date parsing for analysis
    
    #dataLoadStartDate = dataLoadEndDate - relativedelta(years=years_to_load)
    #print(dataLoadStartDate)
    set_segment_and_load_signal(issue, years_to_load, segments, dataLoadEndDate, beLongThreshold)
