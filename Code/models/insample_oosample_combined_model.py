# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

insample_sample_model.py
"""
import sys
import json

sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import PlotUtility
from time_utils import TimeUtility
from retrieve_data import DataRetrieve, ComputeTarget
from candle_indicators import CandleIndicators
from transformers import Transformers
from ta_momentum_studies import TALibMomentumStudies
from model_utils import ModelUtility
from feature_generator import FeatureGenerator
from config import current_feature, feature_dict
import models_utils

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os.path
import pickle
#from pandas.tseries.holiday import USFederalHolidayCalendar
#from pandas.tseries.offsets import CustomBusinessDay
#us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    plotIt = PlotUtility()
    timeUtil = TimeUtility()
    ct = ComputeTarget()
    candle_ind = CandleIndicators()
    dSet = DataRetrieve()
    taLibMomSt = TALibMomentumStudies()
    transf = Transformers()
    modelUtil = ModelUtility()
    featureGen = FeatureGenerator()
        
    issue = "TLT"
    # Set IS-OOS parameters
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 2
    oos_months = 3
    segments = 4
    
    dataSet = dSet.read_issue_data(issue)
    
    # get first data from loaded data instead of hard coding start date
    dataSet = dSet.set_date_range(dataSet, "2014-09-26", pivotDate)
    
    #set beLong level
    beLongThreshold = 0.000
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)

    input_dict = {} # initialize
    input_dict = {'f1': 
              {'fname' : 'ATR', 
               'params' : [5],
               'transform' : ['Zscore', 10]
               },
              'f2': 
              {'fname' : 'RSI', 
               'params' : [2],
               'transform' : ['Scaler', 'robust']
               },
              'f3': 
              {'fname' : 'DeltaATRRatio', 
               'params' : [2, 10],
               'transform' : ['Scaler', 'robust']
               }
             }    
    dataSet2 = featureGen.generate_features(dataSet, input_dict)
    
    # save Dataset of analysis
    print("====Saving dataSet====\n\n")
    modelname = 'RF'
    file_title = issue + "_insample_model_" + modelname + ".pkl"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    dataSet.to_pickle(file_name)
    
    # set date splits
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    is_end_date = isOosDates[4]
    oos_end_date = isOosDates[5]
    
    modelStartDate = is_start_date
    modelEndDate = modelStartDate + relativedelta(months=is_months)
    print("Issue: " + issue)
    print("Start date: " + str(modelStartDate) + "  End date: " + str(modelEndDate))

    model_results = []
    predictor_vars = "Temp holding spot"
    
    # IS only
    for i in range(segments):        
        mmData = dataSet[modelStartDate:modelEndDate]
        nrows = mmData.shape[0]
        
        plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
        plotIt.plot_v2x(mmData, plotTitle)
        plotIt.histogram(mmData['beLong'], x_label="beLong signal", y_label="Frequency", 
          title = "beLong distribution for " + issue)        
        plt.show(block=False)

        col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
        to_drop = ['Open','High','Low', 'gainAhead', 'Symbol', 'Date', 'Close']
        for x in to_drop:
            col_vals.append(x)
        mmData = dSet.drop_columns(mmData, col_vals)
        
        models_utils.correlation_matrix(mmData)
        
        names = mmData.columns.values.tolist()            

        with open('columns_to_keep.json', 'w') as outfile:
            json.dump(names, outfile)

        ######################
        # ML section
        ######################
        #  Make 'iterations' index vectors for the train-test split
        iterations = 100
             
        dX, dy = modelUtil.prepare_for_classification(mmData)        
        
        sss = StratifiedShuffleSplit(n_splits=iterations,
                                     test_size=0.33,
                                     random_state=None
                                     )
                
        model = RandomForestClassifier(n_jobs=-1,
                                       random_state=55,
                                       min_samples_split=10,
                                       n_estimators=500,
                                       max_features = 'auto',
                                       min_samples_leaf = 5,
                                       oob_score = 'TRUE'
                                       )
                
        ### add issue, other data OR use dictionary to pass data!!!!!!
        info_dict = {'issue':issue, 'modelStartDate':modelStartDate, 'modelEndDate':modelEndDate, 'modelname':modelname, 'nrows':nrows, 'predictors':predictor_vars}
        model_results = modelUtil.model_and_test(dX, dy, model, model_results, sss, info_dict)
        
        modelStartDate = modelStartDate  + relativedelta(months=oos_months) + BDay(1)
        modelEndDate = modelStartDate + relativedelta(months=is_months) - BDay(1)
        oos_start_date = oos_end_date  + BDay(1)
        oos_end_date = oos_end_date + relativedelta(months=oos_months) - BDay(1)
    
    ## loop ended, print results
    df = pd.DataFrame(model_results)
    df = df[['Issue','StartDate','EndDate','Model','Rows','beLongCount','Predictors','IS-Accuracy','IS-Precision','IS-Recall','IS-F1','OOS-Accuracy','OOS-Precision','OOS-Recall','OOS-F1']]
    print(df)

    ## Save results
    dirext = issue + '_Model_' + modelname + '_start_' + str(dataLoadStartDate) + '_end_' + str(pivotDate) + '_' + datetime.datetime.now().strftime("%Y-%m-%d")
    print(dirext)
    filename = "IS_model_iteration_" + dirext + ".csv"
    current_directory = os.getcwd()
    df.to_csv(current_directory+"\\"+filename, encoding='utf-8', index=False)
    
    # save Dataset
    print("====Saving model====")
    modelname = 'RF'
    file_title = issue + "_predictive_model_" + modelname + ".sav"
    file_name = os.path.join(r'C:\Users\kruegkj\Documents\GitHub\QuantTradingSys\Code\models\model_data', file_title)
    #joblib.dump(model,filename)
    pickle.dump(model, open(file_name, 'wb'))
    
    ############################
    #################################
    ## start of OOS
    # Select the date range
    oos_start_date = isOosDates[2]
    oos_end_date = isOosDates[5]
    
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

        col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
        to_drop = ['Open','High','Low', 'gainAhead', 'Symbol', 'Date', 'Close', 'beLong']
        for x in to_drop:
            col_vals.append(x)
        valModelData = dSet.drop_columns(valData, col_vals)
    
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
