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
import models_utils
from transformers import Transformers

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
import os.path
import pickle
from pandas.tseries.offsets import BDay
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
    segments = 2
    
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
    
    # retrieve dataset from IS
    print("\n====Retrieving dataSet====")
    modelname = 'RF'
    file_title = issue + "_in_sample_" + modelname + ".pkl"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    dataSet = pd.read_pickle(file_name)

    
    # retrieve model IS
    print("====Retrieving model====\n")
    modelname = 'RF'
    file_title = issue + "_predictive_model_" + modelname + ".sav"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    model = pickle.load(open(file_name, 'rb'))
    
    # Select the date range
    modelStartDate = oos_start_date
    modelEndDate = modelStartDate + relativedelta(months=oos_months)
    
    # initialize dataframes for trade analysis
    tradesDataFull = pd.DataFrame()
    valDataFull = pd.DataFrame()
    
    for i in range(segments):
        print("Model start: ", modelStartDate)
        print("Model end: ", modelEndDate)
        
        valData = dataSet[modelStartDate:modelEndDate].copy()
        tradesData = valData.copy()
        
        # Plot price and be lOngs for visual analysis
        models_utils.plotPriceAndBeLong(issue,
                                        modelStartDate,
                                        modelEndDate,
                                        valData
                                        )

        # hack'ish...used to get columns to keep from in sample code
        col_vals = json.load(open('columns_to_keep.json', 'r'))
        col_vals.remove('beLong')
        valModelData = dSet.keep_columns(valData, col_vals)
    
        valRows = valModelData.shape[0]
#        print("There are %i data points" % valRows)
    
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
        plt.plot(equity)
        plt.title('Terminal wealth of predicted trades')
        print("\nTerminal Weatlh: ", equity[valRows-1])
        #TWR is (Final Stake after compounding / Starting Stake) for your system.
        plt.show(block=0)
        
        # Store predictions in valBeLong for plotting
        valData['valBeLong'] = pd.Series(y_validate, index=valData.index)
        
        # Plot price and trading signals
        models_utils.plotPriceAndTradeSignals(valData)       
        
        #Storing info for later trades analysis 
        tradesData['valBeLong'] = pd.Series(y_validate, index=tradesData.index)
        tradesData['gain'] = tradesData['Close'] - tradesData['Open']
        
        #  Count the number of rows in the file
        print('\n\n\nStarting Equity Analysis\n')
        nrows = tradesData.shape[0]
        print ('There are %0.f rows of data' % nrows)
        
        # Getting Cumulative equity values
        models_utils.cumulEquity(valData, tradesData, ceType='All')  
        models_utils.cumulEquity(valData, tradesData, ceType='BeLong')   
        models_utils.cumulEquity(valData, tradesData, ceType='ValBeLong')
        # Plot those values
        models_utils.plotPriceAndCumulEquity(issue, valData)
        
        modelStartDate = modelEndDate  + BDay(1)
        modelStartDate = modelStartDate.date()
        modelEndDate = modelStartDate + relativedelta(months = oos_months) - BDay(1)
        modelEndDate = modelEndDate.date()
        
        tradesDataFull = tradesDataFull.append(tradesData)
        valDataFull = valDataFull.append(valData)
        
     
    
    #  Evaluation of signals      
    print ("\n\n\n\n==========================================")
    print("Trade Analysis\n")
    #  Status variables     
    ndays = tradesDataFull.shape[0]
    
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
        dt = tradesDataFull.index[i]
        dt = dt.date()
        #  Check the date
        # oos_start_date
        # oos_end_date
        datesPass = dt>=oos_start_date and dt<=pivotDate
        if datesPass:
            iTradeDay = iTradeDay + 1
            if sharesHeld[iTradeDay-1] > 0:
                #  In a long position
                if tradesDataFull.valBeLong[i]<0:
                    #  target is -1 -- beFlat 
                    #  Exit -- close the trade
                    exitPrice = tradesDataFull.Close[i]
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
                    MTMPrice = tradesDataFull.Close[i]
                    openTradeEquity = sharesHeld[iTradeDay] * MTMPrice
                    accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                    pass
            else:
                #  Currently flat
                if tradesDataFull.valBeLong[i]>0:
                    #  target is +1 -- beLong
                    #  Enter a new position
                    entryPrice = tradesDataFull.Close[i]
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
        else:
            print("dates don't pass")
    #  Format and print results        
     
    finalAccountBalance = accountBalance[iTradeDay]
    print ('Final account balance: %.2f' %  finalAccountBalance)
    numberTradeDays = iTradeDay        
    numberTrades = iTradeNumber
    print ("Number of trades:", numberTrades)
    
    from pandas import Series
    
    Sequity = Series(accountBalance[0:numberTradeDays-1])
    
    Sequity.plot()
    plt.show(block=False)
    
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
    df_to_save = tradesDataFull[['valBeLong','gainAhead']].copy()
    df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    df_to_save.columns=['Date','signal','gainAhead']
    #print(df_to_save)
    
    dirext = issue + '_test1'
    filename = "oos_equity_eval_" + dirext + ".csv" 
    df_to_save.to_csv(filename, encoding='utf-8', index=False)


    
    print('********************************************************')
    print('\n********************************************************')
    print('\n********************************************************\n\n\n\n')