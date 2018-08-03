# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:17:54 2018

@author: kruegkj

retrieve_data.py
"""


import pandas as pd
import numpy as np
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


class DataRetrieve:   

    def read_issue_data(self, issue):
        self.issue = issue
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
        print ("Successfully retrieved data series for " + issue)
        # Copy Close
        df['Pri'] = df.Close
        return df
    
    def set_date_range(self, df, dfStartDt, dfEndDt):
        us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        df.set_index('Date', inplace=True)
        df3 = df.reindex(pd.date_range(start=dfStartDt, end=dfEndDt, freq=us_cal))
        return df3

    def drop_columns(self, df, col_vals):
        df.drop(col_vals, axis =1, inplace=True)
        return df
    
class ComputeTarget:
    
    def setTarget(self, p, direction, beLongThreshold):
        p['gainAhead'] = ComputeTarget.gainAhead(p.Pri)
        p['beLong'] = np.where(p.gainAhead>beLongThreshold,1,-1)
        return p

    def gainAhead(p):
        # Computes change in the next 1 bar.
        # p, the base series.
        # Return is a numpy array of changes.
        # A change of 1% is 0.01
        # The final value is unknown.  Its value is 0.0.
        nrows = p.shape[0]
        g = np.zeros(nrows)
        for i in range(0,nrows-1):
            g[i] = (p[i+1]-p[i])/p[i]
            # if % change is 0, change to small number
            if (abs(g[i]) < 0.0001):
                g[i] = 0.0001
        return g
        
    def priceChange(self, p):
        nrows = p.shape[0]
        pc = np.zeros(nrows)
        for i in range(1,nrows):
            pc[i] = (p[i]-p[i-1])/p[i-1]
        return pc
        
if __name__ == "__main__":
    from plot_utils import *
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2014-06-01"
    issue = "TLT"
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    #dataSet.set_index('Date', inplace=True)
    
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    beLongThreshold = 0.0
    ct = ComputeTarget()
    targetDataSet = ct.setTarget(dataSet, "Long", beLongThreshold)
    nrows = targetDataSet.shape[0]
    print ("nrows: ", nrows)
    print (targetDataSet.shape)
    print (targetDataSet.tail(10))
    
    targetDataSet = dSet.drop_columns(targetDataSet,['High','Low'])
    
#    df_to_save = targetDataSet.copy()
#    df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
#    df_to_save.to_csv("sample targets.csv", encoding='utf-8', index=False)
    
    print ("beLong counts: ")
    print (targetDataSet['beLong'].value_counts())
    print ("out of ", nrows)
    
    testFirstYear = "2014-04-01"
    testFinalYear = "2014-06-01"
    qtPlot = targetDataSet.ix[testFirstYear:testFinalYear]
    
#    numSubPlots = 2
#    # format the ticks
#    fig, axes = plt.subplots(numSubPlots,1, figsize=(numSubPlots*5,8), sharex=True)
#    
#    axes[0].plot(qtPlot['Close'], label=issue)
#    axes[1].plot(qtPlot['beLong'], label='beLong');
#
#    # Bring subplots close to each other.
#    plt.subplots_adjust(hspace=0.1)
#    
#    #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
#    # Hide x labels and tick labels for all but bottom plot.
#    for ax in axes:
#        ax.label_outer()
#        ax.legend(loc='upper left', frameon=True, fontsize=8)
#        ax.grid(True, which='both')
#        fig.autofmt_xdate()
#        ax.xaxis_date()
#        ax.autoscale_view()
#        ax.grid(b=True, which='major', color='k', linestyle='-')
#        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
#        ax.minorticks_on()
#        ax.tick_params(axis='y',which='minor',bottom='off')
    
    plotIt = PlotUtility()
    
    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(qtPlot['Pri'], plotTitle)
    
    plotTitle = issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v2x(qtPlot['Pri'], qtPlot['beLong'], plotTitle)