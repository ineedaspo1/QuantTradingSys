# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:35:32 2019

@author: KRUEGKJ

OOS day by day
"""

# Import standard libraries
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os
import os.path
import pickle
import random
import json
import sys

from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Import custom libraries
from Code.lib.plot_utils import PlotUtility
from Code.lib.time_utils import TimeUtility
from Code.lib.retrieve_data import DataRetrieve, ComputeTarget
from Code.lib.retrieve_system_info import TradingSystemUtility
from Code.lib.candle_indicators import CandleIndicators
from Code.lib.transformers import Transformers
from Code.lib.ta_momentum_studies import TALibMomentumStudies
from Code.lib.model_utils import ModelUtility, TimeSeriesSplitImproved
from Code.lib.feature_generator import FeatureGenerator
from Code.utilities.stat_tests import stationarity_tests
from Code.lib.config import current_feature, feature_dict
from Code.models import models_utils
from Code.lib.model_algos import AlgoUtility

plotIt = PlotUtility()
timeUtil = TimeUtility()
ct = ComputeTarget()
candle_ind = CandleIndicators()
dSet = DataRetrieve()
taLibMomSt = TALibMomentumStudies()
transf = Transformers()
modelUtil = ModelUtility()
featureGen = FeatureGenerator()
dSet = DataRetrieve()
modelAlgo = AlgoUtility()
sysUtil = TradingSystemUtility()

if __name__ == '__main__':
       
    # set to existing system name OR set to blank if creating new
    if len(sys.argv) < 2:
        print('You must set a system_name or set to """"!!!')
    
    system_name = sys.argv[1]
    best_model_name = sys.argv[2]
    best_model_segment = sys.argv[3]
    
    system_directory = sysUtil.get_system_dir(system_name)
    #ext_input_dict = sys.argv[2]
    
    print("Existing system")
    
    # Get info from system_dict
    system_dict = sysUtil.get_dict(system_directory, 'system_dict')
    issue = system_dict["issue"]
    
    is_oos_ratio = system_dict["is_oos_ratio"]
    oos_months = system_dict["oos_months"]
    segments = system_dict["segments"]
    
    system_dict["best_model_name"] = best_model_name
    system_dict["best_model_segment"] = best_model_segment
    sysUtil.save_dict(system_name, 'system_dict', system_dict)
    
    print(system_dict)
    
    # get feature list
    feature_dict = sysUtil.get_dict(system_directory, 'feature_dict')
    
    # Set IS-OOS parameters
    pivotDate = system_dict['pivotDate']
#    pivotDate = datetime.strptime(pivotDate, '%Y-%m-%d')
    new_pivot_date = datetime.datetime.strptime(pivotDate, '%Y-%m-%d').date()

    # set date splits
    isOosDates = timeUtil.is_oos_data_split(issue, new_pivot_date, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    is_end_date = isOosDates[4]
    oos_end_date = isOosDates[5]
    
    modelStartDate = oos_start_date
    modelEndDate = modelStartDate + relativedelta(months=oos_months)
    print("Issue: " + issue)
    print("OOS Start date: " + str(modelStartDate) + "  OOS End date: " + str(modelEndDate))
    
    # get raw dataset
    file_title = "raw-features-" + system_name + ".pkl"
    file_name = os.path.join(system_directory, file_title)
    dataSet = pd.read_pickle(file_name)
    
    dataSet.head(5)
    
    # Retrieve best model
    # Best model should be updated in system_dict
    # If not, ask for input of model 
    # Also need to account for segment in title
    #best_model_name = system_dict["best_model"]
    file_title = "fit-model-" + best_model_name + "-IS-" + system_name + "-" + best_model_segment +".sav"
    file_name = os.path.join(system_directory, file_title)
    model = pickle.load(open(file_name, 'rb'))
    
    # Initialize dataframes for trade analysis
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
        
        col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
        to_drop = ['Open','High','Low', 'gainAhead', 'Close', 'Volume', 'AdjClose', 'beLong']
        for x in to_drop:
            col_vals.append(x)
        valModelData = dSet.drop_columns(valData, col_vals)
    
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
        datesPass = dt>=oos_start_date and dt<=new_pivot_date
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
    Sequity.plot(grid=True, title="Equity Curve", )
    plt.show(block=False)
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
    ####  end and save Shadow trades  ####     
    df_to_save = tradesDataFull[['valBeLong','gainAhead','Close']].copy()
    df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    df_to_save.columns=['Date','signal','gainAhead','Close']
    #print(df_to_save)
    dSet.save_csv(system_directory,
                  system_name,
                  'OOS_Equity',
                  'new', 
                  df_to_save
                  )
    
    dSet.save_csv(system_directory,
                  system_name,
                  'OOS_Equity',
                  'dbd', 
                  df_to_save
                  )

    print(df_to_save.tail(10))
    
    print('********************************************************')
    print('\n********************************************************')
    print('\n********************************************************\n\n\n\n')

    