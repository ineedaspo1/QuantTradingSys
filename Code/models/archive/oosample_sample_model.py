# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:11:34 2018

@author: KRUEGKJ
"""
import sys
import json
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import PlotUtility
from time_utils import TimeUtility
from retrieve_data import DataRetrieve, ComputeTarget
from transformers import Transformers

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
import os.path
import pickle

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

if __name__ == "__main__":
    
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    ct = ComputeTarget()
    dSet = DataRetrieve()
 
    #set beLong level
    beLongThreshold = 0.000
    
    issue = "TLT"
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 4
    oos_months = 3
    segments = 3
    
    isOosDates = timeUtil.is_oos_data_split(issue,
                                            pivotDate,
                                            is_oos_ratio,
                                            oos_months,
                                            segments
                                            )
    dataLoadStartDate   = isOosDates[0]
    is_start_date       = isOosDates[1]
    oos_start_date      = isOosDates[2]
    is_months           = isOosDates[3]
    is_end_date         = isOosDates[4]
    oos_end_date        = isOosDates[5]
    
    # Select the date range
    modelStartDate = oos_start_date
    print("Model start: ", modelStartDate)
    modelEndDate = modelStartDate + relativedelta(months=oos_months)
    print("Model end: ", modelEndDate)


    # retrieve dataset from IS
    print("\n\n====Retrieving dataSet====")
    modelname = 'RF'
    file_title = issue + "_in_sample_" + modelname + ".pkl"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    dataSet = pd.read_pickle(file_name)
    #dataSet.set_index('Date', inplace=True)
    
    # retrieve model IS
    print("\n\n====Retrieving model====")
    modelname = 'RF'
    file_title = issue + "_predictive_model_" + modelname + ".sav"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    model = pickle.load(open(file_name, 'rb'))
    
#    valData = dataSet.reindex(df2)
    valData = dSet.set_date_range(dataSet, modelStartDate, modelEndDate)
    tradesData = valData
        
#    nrows = valData.shape[0]
#    print ("\n\nbeLong counts: ")
#    be_long_count = valData['beLong'].value_counts()
#    print (be_long_count)
#    print ("out of ", nrows)
    
    plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
    plotIt.plot_v2x(valData, plotTitle)
    plotIt.histogram(valData['beLong'], x_label="beLong signal", y_label="Frequency", 
      title = "beLong distribution for " + issue)        
    plt.show(block=False)
    
    col_vals = json.load(open('columns_to_keep.json', 'r'))
    col_vals.remove('beLong')
    valModelData = dSet.keep_columns(valData, col_vals)
    
    valRows = valModelData.shape[0]
    print("There are %i data points" % valRows)
    
    #dy = np.zeros_like(valModelData)
    #dy = valModelData.values
    
    # test the validation data
    y_validate = []
    y_validate = model.predict(valModelData)
    #y_validate = model.predict(dy)
    
    # Create best estimate of trades
    bestEstimate = np.zeros(valRows)
    
    # You may need to adjust for the first and / or final entry 
    for i in range(valRows -1):
        print(valData.gainAhead.iloc[i], y_validate[i])
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
    plt.plot(equity)
    
    print("\n End of Run")
    
    valData['valBeLong'] = pd.Series(y_validate, index=valData.index)
    
    
    #==========================
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
    ax1.plot(valData['Close'])
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
    ax2.plot(valData.Close, color='black',alpha=0.6,label='CLOSE',linestyle='--')
    ax2.legend(loc='center left', frameon=True, fontsize=8)
    ax2.label_outer()
    plt.show()
    
    #===================================        
    # Getting sample equity curve
    
    
    
    
    #  Evaluation of signals
    
    print ("Starting single run")
    
    #  Status variables
    
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
                    exitPrice = tradesData.Close[i]
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
                    MTMPrice = tradesData.Close[i]
                    openTradeEquity = sharesHeld[iTradeDay] * MTMPrice
                    accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                    pass
            else:
                #  Currently flat
                if tradesData.valBeLong[i]>0:
                    #  target is +1 -- beLong
                    #  Enter a new position
                    entryPrice = tradesData.Close[i]
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
    print ('Final account balance: %.2f' %  finalAccountBalance)
    numberTradeDays = iTradeDay        
    numberTrades = iTradeNumber
    print ("Number of trades:", numberTrades)
    
    from pandas import Series
    
    Sequity = Series(accountBalance[0:numberTradeDays-1])
    
    Sequity.plot()
    
    tradeWins = sum(1 for x in tradeGain if float(x) >= 1.0)
    tradeLosses = sum(1 for x in tradeGain if float(x) < 1.0 and float(x) > 0)
    print("Wins: ", tradeWins)
    print("Losses: ", tradeLosses)
    print("W/L: ", tradeWins/numberTrades)
    
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
    print(df_to_save)
    
    dirext = issue + '_test1'
    filename = "oos_equity_eval_" + dirext + ".csv" 
    df_to_save.to_csv(filename, encoding='utf-8', index=False)
