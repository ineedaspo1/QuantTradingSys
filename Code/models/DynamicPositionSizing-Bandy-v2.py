from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:59:06 2018

@author: kruegkj
"""

"""
Figure 9.21
DynamicPositionSizing.py

This program is included in the book
"Quantitative Technical Analysis"
written by Dr. Howard Bandy
and published by Blue Owl Press, Inc.
Copyright 2014 Howard Bandy

Author: Howard Bandy
Blue Owl Press, Inc.
www.BlueOwlPress.com

This program is intended to be an educational tool.
It has not been reviewed.
It is not guaranteed to be error free.
Use of any kind by any person or organization
is with the understanding that the program is as is,
including any and all of its faults.
It is provided without warranty of any kind.
No support for this program will be provided.
It is not trading advice.
The programming is intended to be clear,
but not necessarily efficient.

Permission to use, copy, and share is granted.
Please include author credits with any distribution.

"""
import sys
sys.path.append('../lib')
sys.path.append('../utilities')
from plot_utils import *

plotIt = PlotUtility()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from pandas.core import datetools
import time

issue = 'TLT'
#  Set the path for the csv file
#path = 'oos_equity_eval_EWH_test1.csv'
# latest RT plus OOS
path = 'TLT_tms1_2018-04-02.csv'

#  Use pandas to read the csv file, 
#  creating a pandas dataFrame
sst = pd.read_csv(path)
#print type(sst)

#  Print the column labels
#print sst.columns.values
#print sst.head()
#print sst.tail()

#  Count the number of rows in the file
nrows = sst.shape[0]
#print 'There are %0.f rows of data' % nrows

#  Compute cumulative equity for all days
#equityAllSignals = np.zeros(nrows)
#equityAllSignals[0] = 1
#for i in range(1,nrows):
#    equityAllSignals[i] = (1+sst.gainAhead[i])*equityAllSignals[i-1]
#
#print 'TWR for all signals is %0.3f' % equityAllSignals[nrows-1]
    
#  Compute cumulative equity for days with beLong signals    
#equityBeLongSignals = np.zeros(nrows)
#equityBeLongSignals[0] = 1
#for i in range(1,nrows):
#    if (sst.signal[i] > 0):
#        equityBeLongSignals[i] = (1+sst.gainAhead[i])*equityBeLongSignals[i-1]
#    else:
#        equityBeLongSignals[i] = equityBeLongSignals[i-1]
#        
#print 'TWR for all days with beLong signals is %0.3f' % equityBeLongSignals[nrows-1]

#  Plot the two equity streams
#plt.plot(equityBeLongSignals, '.')
#plt.plot(equityAllSignals, '--')
#plt.show()

sst = sst.set_index(pd.DatetimeIndex(sst['Date']))
sst=sst.drop('Date', axis=1)
sst['safef'] = 0.0
sst['CAR25'] = 0.0

#print sst.columns.values
#print sst.head()
#print sst.tail()

# create a range of times
#  start date is inclusive
#start = dt.datetime(2010,1,4)
start = dt.datetime(2017,6,2)
#  end date is inclusive
end = dt.datetime(2018,4,2)
updateInterval = 1

forecastHorizon = 190
initialEquity = 100000
ddTolerance = 0.10
tailRiskPct = 95
windowLength = 1*forecastHorizon
nCurves = 50

years_in_forecast = forecastHorizon / 252.0

#  Work with the index rather than the date    
iStart = sst.index.get_loc(start)
print(iStart)
iEnd = sst.index.get_loc(end)
print(iEnd)

printDetails = False

for i in range(iStart, iEnd+1, updateInterval):
    if printDetails: 
        print ("\nDate: ", dt.datetime.strftime(sst.index[i], '%Y-%m-%d'))
        print ("beLong: ", sst.signal[i])
        print ("gain Ahead: {0:.4f}".format(sst.gainAhead[i]))

#  Initialize variables
    curves = np.zeros(nCurves)
    numberDraws = np.zeros(nCurves)
    TWR = np.zeros(nCurves)
    maxDD = np.zeros(nCurves)
    
    fraction = 1.00
    dd95 = 2 * ddTolerance
    
    while (abs(dd95-ddTolerance)>0.03):
        #  Generate nCurve equity curves
        if printDetails: 
            print  ("    Fraction {0:.2f}".format(fraction))
#    
        for nc in range(nCurves):
            #print ("working on curve ", nc)
            equity = initialEquity
            maxEquity = equity
            drawdown = 0
            maxDrawdown = 0
            horizonSoFar = 0
            nd = 0
            while (horizonSoFar < forecastHorizon):
                j = np.random.randint(0,windowLength)
        #        print j
                nd = nd + 1
                weightJ = 1.00 - j/windowLength
        #        print weightJ
                horizonSoFar = horizonSoFar + weightJ
                signalJ = sst.signal[i-j]
                if signalJ > 0:
                    tradeJ = sst.gainAhead[i-j] * weightJ
                else:
                    tradeJ = 0.0
                thisTrade = fraction * tradeJ * equity    
                equity = equity + thisTrade
                maxEquity = max(equity,maxEquity)
                drawdown = (maxEquity-equity)/maxEquity
                maxDrawdown = max(drawdown,maxDrawdown)
    #        print "equity, maxDD, ndraws:", equity, maxDrawdown, nd        
            TWR[nc] = equity
            maxDD[nc] = maxDrawdown
            numberDraws[nc] = nd
    
        #  Find the drawdown at the tailLimit-th percentile        
        dd95 = stats.scoreatpercentile(maxDD,tailRiskPct)
        if printDetails: 
            print ('  DD {0}: {1:.3f} '.format(tailRiskPct, dd95))
        fraction = fraction * ddTolerance / dd95
        TWR25 = stats.scoreatpercentile(TWR,25)        
        CAR25 = 100*(((TWR25/initialEquity) ** (1.0/years_in_forecast))-1.0)
    if printDetails: 
        print ('Fraction: {0:.2f}'.format(fraction))
        print ('CAR25: {0:.2f}'.format(CAR25))
    sst.iloc[i,sst.columns.get_loc('safef')] = fraction
    sst.iloc[i,sst.columns.get_loc('CAR25')] = CAR25
    #sst.loc[i,'CAR25'] = CAR25

print ("Max DD: {}".format(maxDD))        
print ("Number of draws: {}".format(numberDraws))

print (sst.tail(10))
df_to_save = sst.copy()
df_to_save.reset_index(level=df_to_save.index.names, inplace=True)

dirext = issue + 'DynamicRunPartA_1yr_WinLength_' + str(windowLength)
filename = dirext + ".csv" 
df_to_save.to_csv(filename, encoding='utf-8', index=False)
print ("Writing to disk in csv format")


# -----------------------------------------
#
#  compute equity, maximum equity, drawdown, and maximum drawdown
sst1 = sst.copy()
sst1['trade'] = 0.0
sst1['fract'] = 0.0
sst1['equity'] = 0.0
sst1['maxEquity'] = 0.0
sst1['drawdown'] = 0.0
sst1['maxDD'] = 0.0

initialEquity = 100000

#sst1.safef[0] = 1.0
sst1.iloc[0,sst1.columns.get_loc('safef')] = 1.0
#sst1.CAR25[0] = 10.0
sst1.iloc[0,sst1.columns.get_loc('CAR25')] = 10.0
#sst1.equity[0] = initialEquity
sst1.iloc[0,sst1.columns.get_loc('equity')] = initialEquity

for i in range(1,nrows):
    if (sst1.iloc[i,sst1.columns.get_loc('safef')]==0 and sst1.iloc[i,sst1.columns.get_loc('CAR25')]==0):
        sst1.iloc[i,sst1.columns.get_loc('safef')] = sst1.iloc[i-1,sst1.columns.get_loc('safef')]
        sst1.iloc[i,sst1.columns.get_loc('CAR25')] = sst1.iloc[i-1,sst1.columns.get_loc('CAR25')]
        #sst1.CAR25[i] = sst1.CAR25[i-1]
        sst1.iloc[i,sst1.columns.get_loc('fract')] = sst1.iloc[i,sst1.columns.get_loc('safef')]
        #sst1.fract[i] = sst1.safef[i]
        sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')]
    else:
        #sst1.fract[i] = sst1.safef[i]
        sst1.iloc[i,sst1.columns.get_loc('fract')] = sst1.iloc[i,sst1.columns.get_loc('safef')]
        #st1.equity[i] = sst1.equity[i-1]
        sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')]
        
for i in range(iStart, iEnd):
    if (sst1.signal[i] > 0):
        #sst1.trade[i] = sst1.fract[i-1] * sst1.equity[i-1] * sst1.gainAhead[i]
        sst1.iloc[i,sst1.columns.get_loc('trade')] = sst1.iloc[i-1,sst1.columns.get_loc('fract')] * sst1.iloc[i-1,sst1.columns.get_loc('equity')] * sst1.iloc[i,sst1.columns.get_loc('gainAhead')]
    else:
        #sst1.trade[i] = 0.0
        sst1.iloc[i,sst1.columns.get_loc('trade')] = 0.0
    #sst1.equity[i] = sst1.equity[i-1] + sst1.trade[i]
    sst1.iloc[i,sst1.columns.get_loc('equity')] = sst1.iloc[i-1,sst1.columns.get_loc('equity')] + sst1.iloc[i,sst1.columns.get_loc('trade')]
    #sst1.maxEquity[i] = max(sst1.equity[i],sst1.maxEquity[i-1])
    sst1.iloc[i,sst1.columns.get_loc('maxEquity')] = max(sst1.iloc[i,sst1.columns.get_loc('equity')],sst1.iloc[i-1,sst1.columns.get_loc('maxEquity')])
    #sst1.drawdown[i] = (sst1.maxEquity[i]-sst1.equity[i]) / sst1.maxEquity[i]
    sst1.iloc[i,sst1.columns.get_loc('drawdown')] = (sst1.iloc[i,sst1.columns.get_loc('maxEquity')] - sst1.iloc[i,sst1.columns.get_loc('equity')]) / sst1.iloc[i,sst1.columns.get_loc('maxEquity')]
    #sst1.maxDD[i] = max(sst1.drawdown[i],sst1.maxDD[i-1])
    sst1.iloc[i,sst1.columns.get_loc('maxDD')] =  max(sst1.iloc[i,sst1.columns.get_loc('drawdown')],sst1.iloc[i-1,sst1.columns.get_loc('maxDD')])

print (sst1.head(10))
print (sst1.tail(10))

#  Plot the equitycurve and drawdown

#plt.subplot(2,1,1)
#plt.plot(sst1.equity[:-2])
plotTitle = "Equity curve for  " + issue + ", " + str(start) + " to " + str(end)
plotIt.plot_v1(sst1['equity'][:-2], plotTitle)
#plt.subplot(2,1,2)
#plt.plot(sst1.drawdown[:-2])
plotTitle = "Drawdown for  " + issue + ", " + str(start) + " to " + str(end)
plotIt.plot_v1(sst1['drawdown'][:-2], plotTitle)
plt.show

# saving content
df_to_save = sst1.copy()
df_to_save.reset_index(level=df_to_save.index.names, inplace=True)

dirext = issue + 'DynamicRunPartB_1yr_WinLength_' + str(windowLength)
filename = dirext + ".csv" 
df_to_save.to_csv(filename, encoding='utf-8', index=False)

#  /////  end  /////