# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:11:34 2018

@author: KRUEGKJ

Goals:
1. Run all OOS to live trade analysis in one code snippet to better 
understand the entire process and make code improvements when building
related functions to complete this work in an automated manner.

Steps:
1. Run IS analysis in separate program. Use the transformed dataSet and
model for this analysis.
2. 

"""

import sys
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import *
from retrieve_data import *
from indicators import *
from transformers import *
from time_utils import *

import pandas as pd
import numpy as np
#import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os.path
import pickle
from scipy import stats
import datetime
#from pandas.core import datetools
#import time

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

def retrieve_dataSet(issue, model):
    # retrieve dataset from IS
    print("====Retrieving dataSet====")
    file_title = issue + "_in_sample_" + model + ".pkl"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    dataSet = pd.read_pickle(file_name)
    dataSet.set_index('Date', inplace=True)
    return dataSet

def retrieve_model(issue, model):
    # retrieve model IS
    print("====Retrieving model====")
    file_title = issue + "_predictive_model_" + model + ".sav"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    model = pickle.load(open(file_name, 'rb'))
    return model

if __name__ == "__main__":
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    ct = ComputeTarget()
 
    #set beLong level
    beLongThreshold = 0.000
    
    issue = "TLT"
    model = "RF"
    
    pleasePlot = False
    
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 3
    oos_months = 8
    segments = 1
    
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    
    is_end_date = is_start_date + relativedelta(months=is_months)
    oos_end_date = oos_start_date + relativedelta(months=oos_months)
    
    # Select the date range
    modelStartDate = oos_start_date
    print("Model start: ", modelStartDate)
    modelEndDate = modelStartDate + relativedelta(months=oos_months)
    print("Model end: ", modelEndDate)

    dataSet = retrieve_dataSet(issue, model)
    model = retrieve_model(issue, model)
    
    
    df2 = pd.date_range(start=modelStartDate, end=modelEndDate, freq=us_cal)
    valData = dataSet.reindex(df2)
    tradesData = valData
        
    nrows = valData.shape[0]
    print ("beLong counts: ")
    be_long_count = valData['beLong'].value_counts()
    print (be_long_count)
    print ("out of ", nrows)
    
    valData = valData.drop(['Open','High','Low','Close', 'Symbol','percReturn'],axis=1)
    
    if pleasePlot:
        plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
        plotIt.plot_v2x(valData['Pri'], valData['beLong'], plotTitle)
        plotIt.histogram(valData['beLong'], x_label="beLong signal", y_label="Frequency", 
          title = "beLong distribution for " + issue)        
        plt.show(block=False)
    
    valModelData = valData.drop(['Pri','beLong','gainAhead'],axis=1)
    
    valRows = valModelData.shape[0]
    print("There are %i data points" % valRows)
    
    # test the validation data
    y_validate = []
    y_validate = model.predict(valModelData)
    
    # Create best estimate of trades
    bestEstimate = np.zeros(valRows)
    
    # You may need to adjust for the first and / or final entry 
    for i in range(valRows -1):
        #print(valData.gainAhead.iloc[i], y_validate[i])
        if y_validate[i] > 0.0: 
            bestEstimate[i] = valData.gainAhead.iloc[i]
        else:
            bestEstimate[i] = 0.0 
            
    # Create and plot equity curve
    equity = np.zeros(valRows)
    equity[0] = 1.0
    for i in range(1,valRows):
        equity[i] = (1+bestEstimate[i])*equity[i-1]
        
    print("\nTerminal Weatlh: ", equity[valRows-1])
    #plt.plot(equity)
    
    valData['valBeLong'] = pd.Series(y_validate, index=valData.index)
    
    #==========================
    if pleasePlot:
        import matplotlib as mpl
        #from matplotlib import style
        plt.style.use('seaborn-ticks')
        import matplotlib.ticker as ticker
        
        fig = plt.figure(figsize=(12,8))
        plt.subplots_adjust(hspace=0.05)
        ax1 = plt.subplot2grid((6,1), (0,0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid((6,1), (4,0), rowspan=1, colspan=1)
        ax3 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1)
        
        ax2.plot(valData['valBeLong'], color='green', alpha =0.6)
        ax1.plot(valData['Pri'])
        ax3.plot(valData['beLong'], color='purple', alpha =0.6)
        
        ax1.label_outer()
        ax2.label_outer()
        ax2.tick_params(axis='x',which='major',bottom='on')
        ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
        ax2.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
        ax3.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
        ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax2.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax3.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()
        ax1.label_outer()
        ax1.legend(loc='upper left', frameon=True, fontsize=8)
        ax2.label_outer()
        ax2.legend(loc='upper left', frameon=True, fontsize=8)
        ax3.label_outer()
        ax3.legend(loc='upper left', frameon=True, fontsize=8)
    
    #==========================
    tradesData['valBeLong'] = pd.Series(y_validate, index=tradesData.index)
    tradesData['gain'] = tradesData['Close'] - tradesData['Open']
    
    #  Count the number of rows in the file
    nrows = tradesData.shape[0]
    print ('There are %0.f rows of data' % nrows)
    
    #  Compute cumulative equity for all days
    equityAllSignals = np.zeros(nrows)
    equityAllSignals[0] = 1
    for i in range(1,nrows):
        equityAllSignals[i] = (1+tradesData.gainAhead[i])*equityAllSignals[i-1]
    
    print ('TWR for all signals is %0.3f' % equityAllSignals[nrows-1])
    # add to valData
    valData['equityAllSignals'] = pd.Series(equityAllSignals, index=valData.index)
        
    #  Compute cumulative equity for days with beLong signals    
    equityBeLongSignals = np.zeros(nrows)
    equityBeLongSignals[0] = 1
    for i in range(1,nrows):
        if (tradesData.beLong[i] > 0):
            equityBeLongSignals[i] = (1+tradesData.gainAhead[i])*equityBeLongSignals[i-1]
        else:
            equityBeLongSignals[i] = equityBeLongSignals[i-1]
    valData['equityBeLongSignals'] = pd.Series(equityBeLongSignals, index=valData.index)
    
    #  Compute cumulative equity for days with Validation beLong signals    
    equityValBeLongSignals = np.zeros(nrows)
    equityValBeLongSignals[0] = 1
    for i in range(1,nrows):
        if (tradesData.valBeLong[i] > 0):
            equityValBeLongSignals[i] = (1+tradesData.gainAhead[i])*equityValBeLongSignals[i-1]
        else:
            equityValBeLongSignals[i] = equityValBeLongSignals[i-1]
                   
    print ('TWR for all days with beLong signals is %0.3f' % equityBeLongSignals[nrows-1])
    valData['equityValBeLongSignals'] = pd.Series(equityValBeLongSignals, index=valData.index)
    
    #  Plot the two equity streams
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(issue + ' Portfolio value in Validation')
    ax1 = fig.add_subplot(111)
    ax1.plot(valData.equityBeLongSignals, color='green',label='BeLong')
    ax1.plot(valData.equityAllSignals, color='blue',label='Equity')
    ax1.plot(valData.equityValBeLongSignals, color='purple',label='ValBeLong')
    
    ax1.legend(loc='upper left', frameon=True, fontsize=8)
    ax1.label_outer()
    ax1.tick_params(axis='x',which='major',bottom='on')
    ax1.minorticks_on()
    ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
    ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
    
    ax2 = ax1.twinx()
    ax2.plot(valData.Pri, color='black',alpha=0.6,label='CLOSE',linestyle='--')
    ax2.legend(loc='center left', frameon=True, fontsize=8)
    ax2.label_outer()
    plt.show()
    
    #===================================        
    # Getting sample equity curve
    #  Evaluation of signals
    print ("Starting single run")
    #  Status variable
    ndays = tradesData.shape[0]
    #  Local variables for trading system
    initialEquity = 100000
    fixedTradeDollars = 10000
    commission = 0.005      #  Dollars per share per trade
    #  These are scalar and apply to the current conditions
    entryPrice = 0
    exitPrice = 0
    #  These have an element for each day loaded
    #  Some will be unnecessary
    accountBalance = np.zeros(ndays)
    cash = np.zeros(ndays)
    sharesHeld = np.zeros(ndays)
    tradeGain = []
    tradeGainDollars = []
    openTradeEquity = np.zeros(ndays)
    tradeWinsValue = np.zeros(ndays)
    tradeLossesValue = np.zeros(ndays)
    
    iTradeDay = 0
    iTradeNumber = 0
    #  Day 0 contains the initial values
    accountBalance[0] = initialEquity
    cash[0] = accountBalance[0]
    sharesHeld[0] = 0
    
    #  Loop over all the days loaded
    for i in range (1,ndays):
        #  Extract the date
        dt = tradesData.index[i]
        #  Check the date
        datesPass = dt.date()>=modelStartDate and dt.date()<=modelEndDate
        if datesPass:
            iTradeDay = iTradeDay + 1
            if sharesHeld[iTradeDay-1] > 0:
                #  In a long position
                if tradesData.valBeLong[i]<0:
                    #  target is -1 -- beFlat 
                    #  Exit -- close the trade
                    exitPrice = tradesData.Pri[i]
                    grossProceeds = sharesHeld[iTradeDay-1] * exitPrice
                    commissionAmount = sharesHeld[iTradeDay-1] * commission
                    netProceeds = grossProceeds - commissionAmount
                    #print("netProceeds: ", netProceeds)
                    cash[iTradeDay] = cash[iTradeDay-1] + netProceeds
                    accountBalance[iTradeDay] = cash[iTradeDay]
                    sharesHeld[iTradeDay] = 0
                    iTradeNumber = iTradeNumber+1
                    #tradeGain[iTradeNumber] = (exitPrice / (1.0 * entryPrice))    
                    tradeGain.append(exitPrice / (1.0 * entryPrice))
                    tradeGainDollars.append(((exitPrice / (1.0 * entryPrice))*fixedTradeDollars)-fixedTradeDollars)
                    
                    pass
                else:
                    #  target is +1 -- beLong
                    #  Continue long
                    sharesHeld[iTradeDay] = sharesHeld[iTradeDay-1]
                    cash[iTradeDay] = cash[iTradeDay-1]
                    MTMPrice = tradesData.Pri[i]
                    openTradeEquity = sharesHeld[iTradeDay] * MTMPrice
                    accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                    pass
            else:
                #  Currently flat
                if tradesData.valBeLong[i]>0:
                    #  target is +1 -- beLong
                    #  Enter a new position
                    entryPrice = tradesData.Pri[i]
                    sharesHeld[iTradeDay] = int(fixedTradeDollars/(entryPrice+commission))
                    shareCost = sharesHeld[iTradeDay]*(entryPrice+commission)
                    cash[iTradeDay] = cash[iTradeDay-1] - shareCost
                    openTradeEquity = sharesHeld[iTradeDay]*entryPrice
                    accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                    pass
                else:
                    #  target is -1 -- beFlat
                    #  Continue flat
                    cash[iTradeDay] = cash[iTradeDay-1]
                    accountBalance[iTradeDay] = cash[iTradeDay] 
                    pass
                
    #  Format and print results        
    finalAccountBalance = accountBalance[iTradeDay]
    valData['accountBalance'] = pd.Series(accountBalance, index=valData.index)
    
    plotIt.plot_v1(valData['accountBalance'], "Equity curve")
    print ('Final account balance: %.2f' %  finalAccountBalance)
    numberTradeDays = iTradeDay        
    numberTrades = iTradeNumber
    print ("Number of trades:", numberTrades)
    
    from pandas import Series
    
    Sequity = Series(accountBalance[0:numberTradeDays-1])

    
    tradeWins = sum(1 for x in tradeGain if float(x) >= 1.0)
    tradeLosses = sum(1 for x in tradeGain if float(x) < 1.0 and float(x) > 0)
    print("Wins: ", tradeWins)
    print("Losses: ", tradeLosses)
    print("W/L: %.2f" % (tradeWins/numberTrades))
    
    tradeWinsValue = sum((x*fixedTradeDollars)-fixedTradeDollars for x in tradeGain if float(x) >= 1.0)
    tradeLossesValue = sum((x*fixedTradeDollars)-fixedTradeDollars for x in tradeGain if float(x) < 1.0 and float(x) > 0)
    print('Total value of Wins:  %.2f' % tradeWinsValue)
    print('Total value of Losses:  %.2f' % tradeLossesValue)
    #(Win % x Average Win Size) – (Loss % x Average Loss Size)
    print('Expectancy:  %.2f' % ((tradeWins/numberTrades)*(tradeWinsValue/tradeWins)-(tradeLosses/numberTrades)*(tradeLossesValue/tradeLosses)))
    print("Fixed trade size: ", fixedTradeDollars)
    
    # Sharpe ratio...probably not correct math
    #import math
    #print(np.mean(tradeGainDollars))
    #print(np.std(tradeGainDollars))
    #print(math.sqrt(numberTradeDays)*(np.mean(tradeGainDollars)/np.std(tradeGainDollars)))
    
    
    ####  end  ####     
    df_to_save = tradesData[['valBeLong','gainAhead']].copy()
    df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    df_to_save.columns=['Date','signal','gainAhead']
    #print(df_to_save)
    
    dirext = issue + '_test1'
    filename = "oos_equity_eval_" + dirext + ".csv" 
    df_to_save.to_csv(filename, encoding='utf-8', index=False)
    
    #########################
    print("##############################")
    print("First pass. Caluclating safe-f and CAR25 for ", issue)
    print('\n')      
    
    sst = df_to_save.set_index(pd.DatetimeIndex(df_to_save['Date']))
    sst=sst.drop('Date', axis=1)
    sst['safef'] = 0.0
    sst['CAR25'] = 0.0
    
    # create a range of times
    #  start date is inclusive
    #start = dt.datetime(2010,1,4)
    start = datetime.date(2017, 8, 2)
    #  end date is inclusive
    end = datetime.date(2018, 4, 2)
    
    updateInterval = 1
    
    forecastHorizon = 160
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
    
    if printDetails: 
        print ("Max DD: {}".format(maxDD))        
        print ("Number of draws: {}".format(numberDraws))
    
    print ('\nThe last 5 results:\n')
    print (sst.tail(5))
    df_to_save = sst.copy()
    df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    
    dirext = issue + '_safe-f_CAR25_1yr_WinLength_' + str(windowLength)
    filename = dirext + ".csv" 
    df_to_save.to_csv(filename, encoding='utf-8', index=False)
    print ("Writing to disk in csv format")
    
    
    # -----------------------------------------
    #
    #  compute equity, maximum equity, drawdown, and maximum drawdown
    print ('\n==============================')
    print ('Computing equity, drawdown')
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
    
    #print (sst1.head(10))
    #print (sst1.tail(10))
    
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
    
    dirext = issue + '_eqty_curve_drawdown_1yr_WinLength_' + str(windowLength)
    filename = dirext + ".csv" 
    df_to_save.to_csv(filename, encoding='utf-8', index=False)
    
    #  /////  end  /////
    
    #  Plot the two equity streams
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(issue + ' Portfolio ????')
    ax1 = fig.add_subplot(111)
    ax1.plot(sst1.safef, color='green',label='safe-f')
    ax1.plot(sst1.CAR25, color='blue',label='CAR25')
    #ax1.plot(valData.equityValBeLongSignals, color='purple',label='ValBeLong')
    
    ax1.legend(loc='upper left', frameon=True, fontsize=8)
    ax1.label_outer()
    ax1.tick_params(axis='x',which='major',bottom='on')
    ax1.minorticks_on()
    ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
    ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
    
    sst1['Pri']=valData.Pri
    ax2 = ax1.twinx()
    ax2.plot(sst1.Pri, color='black',alpha=0.6,label='CLOSE',linestyle='--')
    ax2.legend(loc='center left', frameon=True, fontsize=8)
    ax2.label_outer()
    plt.show()
