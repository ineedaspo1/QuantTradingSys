# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
indicators.py


*****************************
TO BE REMOVED FROM CODE BASE****************
******************************
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class Indicators:
    
    def ROC(self,p,lb,feature_dict):
        # Rate of change technical indicator.
        # p, the series having its ROC computed.
        # lb, the lookback period.  Typically 1.
        # Return is a numpy array with values as decimal fractions.
        # A 1% change is 0.01.
        nrows = p.shape[0]
        r = np.zeros(nrows)
        for i in range(lb, nrows):
            r[i] = (p[i]-p[i-lb])/p[i-lb]
        feature_dict['ROC_'+ str(lb)] = 'Keep'
        return r, feature_dict

    def ATR(self,ph,pl,pc,lb,feature_dict):
        # Average True Range technical indicator.
        # ph, pl, pc are the series high, low, and close.
        # lb, the lookback period.  An integer number of bars.
        # True range is computed as a fraction of the closing price.
        # Return is a numpy array of floating point values.
        # Values are non-negative, with a minimum of 0.0.
        # An ATR of 5 points on a issue closing at 50 is
        #    reported as 0.10. 
        nrows = pc.shape[0]
        th = np.zeros(nrows)
        tl = np.zeros(nrows)
        tc = np.zeros(nrows)
        tr = np.zeros(nrows)
        trAvg = np.zeros(nrows)
        
        for i in range(1,nrows):
            if ph[i] > pc[i-1]:
                th[i] = ph[i]
            else:
                th[i] = pc[i-1]
            if pl[i] < pc[i-1]:
                tl[i] = pl[i]
            else:
                tl[i] = pc[i-1]
            tr[i] = th[i] - tl[i]
        for i in range(lb,nrows):
            trAvg[i] = tr[i]            
            for j in range(1,lb-1):
                trAvg[i] = trAvg[i] + tr[i-j]
            trAvg[i] = trAvg[i] / lb
            trAvg[i] = trAvg[i] / pc[i]   
        feature_dict['ATR_'+ str(lb)] = 'Keep'
        return trAvg, feature_dict

    def add_indicators(self,df, ind_list, feature_dict):
        #loop through ind_list
        indDataSet = df
        i = 0
        for i in ind_list:
            print(i)
            sel_ind = i[0]
            if sel_ind == 'RSI':
                indDataSet['RSI'],feature_dict = self.RSI(indDataSet.Pri, i[1], feature_dict)
            elif sel_ind == 'ROC':
                indDataSet['ROC'],feature_dict = self.ROC(indDataSet.Pri, i[1], feature_dict)
            elif sel_ind == 'ATR':
                indDataSet['ATR'],feature_dict = self.ATR(indDataSet.High, indDataSet.Low, indDataSet.Pri, i[1], feature_dict)
            else:
                continue
        return indDataSet, feature_dict

if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}
    
    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
    
    plotIt = PlotUtility()
    
    plotTitle = "Closing price for " + issue + ", " + str(dataLoadStartDate) + " to " + str(dataLoadEndDate)
    plotIt.plot_v1(dataSet['Pri'], plotTitle)
    
    beLongThreshold = 0
    cT = ComputeTarget()
    mmData = cT.setTarget(dataSet, "Long", beLongThreshold)
    
    addIndic1 = Indicators()
    ind_list = [("RSI", 2.3),("ROC",5),("DPO",5),("ATR", 5)]
    dataSet, feature_dict = addIndic1.add_indicators(dataSet, ind_list, feature_dict)
    
    startDate = "2015-02-01"
    endDate = "2015-06-30"
    rsiDataSet = dataSet.ix[startDate:endDate]
    #fig = plt.figure(figsize=(15,8  ))
    fig, axes = plt.subplots(5,1, figsize=(15,8), sharex=True)

    axes[0].plot(rsiDataSet['Pri'], label=issue)
    axes[1].plot(rsiDataSet['RSI'], label='RSI');
    axes[2].plot(rsiDataSet['ROC'], label='ROC');
#    axes[3].plot(rsiDataSet['DPO'], label='DPO');
    axes[4].plot(rsiDataSet['ATR'], label='ATR');
    
    # Bring subplots close to each other.
    plt.subplots_adjust(hspace=0)
    #plt.legend((issue,'RSI','ROC','DPO','ATR'),loc='upper left')
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axes:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=False)
    
    
