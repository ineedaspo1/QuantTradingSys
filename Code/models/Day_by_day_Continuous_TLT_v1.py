# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:34:25 2019

@author: kruegkj

Day-by-day continuous
"""

# Import standard libraries
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os
import os.path
import pickle
import random
import json
import sys
from scipy import stats

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
    #ext_input_dict = sys.argv[2]
    
    print("Existing system")
    
    # Get info from system_dict
    system_directory = sysUtil.get_system_dir(system_name)
    if not os.path.exists(system_directory):
        print("system doesn't exist")
    else:
        file_name = 'system_dict.json'    
        system_dict = dSet.load_json(system_directory, file_name)
        issue = system_dict["issue"]
        direction = system_dict["direction"]
        ver_num = system_dict["ver_num"]
        # Perhaps only load these when needed?
        pivotDate = system_dict["pivotDate"]
        is_oos_ratio = system_dict["is_oos_ratio"]
        oos_months = system_dict["oos_months"]
        segments = system_dict["segments"]
    
    print(system_dict)
    
    # Set IS-OOS parameters
    pivotDate = datetime.strptime(pivotDate, '%Y-%m-%d')
    print('pivotDate: ', pivotDate)
    pivotDate = datetime.date(pivotDate)
    
    # set date splits
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    is_end_date = isOosDates[4]
    oos_end_date = isOosDates[5]
    
    modelStartDate = oos_start_date
    
    # Retrieve feature_dict
    file_name = 'feature_dict.json'
    feature_dict = dSet.load_json(system_directory, file_name)
    
    # get raw data 
    raw_df = dSet.read_issue_data(issue)
    
    # Set trade date
    from datetime import timedelta
    tradeDate = pivotDate + BDay(1)
    print('tradeDate: ', tradeDate)

    if (tradeDate.strftime('%Y-%m-%d') in raw_df['Date'].values):
        print ('date exist')
    else:
        print('no date')
        tradeDate += BDay(1)
        print('tradeDate: ', tradeDate)
        
    # set trade date
    tradeDate = tradeDate.strftime('%Y-%m-%d')
    print(tradeDate)
    
    # set df to trade date range
    trimmed_df = raw_df.iloc[0:raw_df[raw_df.Date == tradeDate].index[0]]
    print(trimmed_df.tail(2))

    price_loc = raw_df.index[raw_df.Date == tradeDate]
    index = price_loc[0]
    print(index)

    # get Close from trade date
    new_open = raw_df.Close[index]
    print(new_open)

    # append price to df
    new_data_df = trimmed_df.append({'Date' : tradeDate , 'Open' : raw_df.Open[index], 'High' : raw_df.High[index], 'Low' : raw_df.Low[index], 'Close' : raw_df.Close[index], 'AdjClose' : raw_df.AdjClose[index], 'Volume' : raw_df.Volume[index] } , ignore_index=True)
    print(new_data_df.tail(2))

    # get first row
    df_start_date = new_data_df.Date[0]
    
    # get last row
    lastRow = new_data_df.shape[0]
    df_end_date = new_data_df.Date[lastRow-1]
    
    feat_df = dSet.set_date_range(new_data_df, df_start_date, df_end_date)
    # Resolve any NA's for now
    feat_df.fillna(method='ffill', inplace=True)
    
    #set beLong level
    beLongThreshold = 0.000
    feat_df = ct.setTarget(new_data_df, "Long", beLongThreshold)
    
    print(feat_df.tail(2))

    # Adding features with new day
    file_name = 'input_dict'
    input_dict = dSet.load_pickle(system_directory, file_name)
    print(input_dict)

    feat_df = featureGen.generate_features(feat_df, input_dict)
    feat_df = transf.normalizer(feat_df, 'Volume', 50)
    
    col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
    to_drop = ['Open','High','Low', 'gainAhead', 'Close', 'Volume', 'AdjClose', 'beLong']
    for x in to_drop:
        col_vals.append(x)
    model_data = dSet.drop_columns(feat_df, col_vals)
    
    # Retrieve model
    best_model_name = "SVM"
    best_model_segment = "segment-0"
    #best_model_name = system_dict["best_model"]
    file_title = "fit-model-" + best_model_name + "-IS-" + system_name + "-" + best_model_segment +".sav"
    file_name = os.path.join(system_directory, file_title)
    model = pickle.load(open(file_name, 'rb'))
    
    # get last row of data
    lastRow = model_data.shape[0]
    model_end_date = model_data.index[lastRow-1]
    print(model_end_date)
    
    # Make prediction
    predict_data = model_data.iloc[lastRow-1]
    dX = np.zeros_like(predict_data)
    dX = predict_data.values
    dX = dX.reshape(1, -1)
    
    # get prediction
    y_validate = []
    y_validate = model.predict(dX)
    y_validate
    print(y_validate[0])
    
    filename = "OOS_Equity_daybyday_" + system_name + ".csv"
    path = system_directory+ "\\" + filename
    shadow_trades = pd.read_csv(path)
    
    # Update shadow trades
    new_st_df = shadow_trades.append({'Date'      : tradeDate,
                                      'signal'    : y_validate[0],
                                      'gainAhead' : 0.000,
                                      'Close'     : feat_df.Close[lastRow-1]
                                      },
                                      ignore_index=True
                                      )
    
    new_st_df['gainAhead'] = ct.gainAhead(new_st_df.Close)
    
    # save updated shadow trades
    filename = "OOS_Equity_daybyday_" + system_name + ".csv"
    new_st_df.to_csv(system_directory+ "\\" + filename, encoding='utf-8', index=False)

    # Load TMS Part 1
    filename = "TMS_Part1_daybyday_" + system_name + ".csv"
    path = system_directory+ "\\" + filename
    tms1 = pd.read_csv(path)
    
    # Update TMS-Part 1 data with latest date
    sst = tms1.append({'Date' : tradeDate , 'signal' : y_validate[0], 'gainAhead' : 0, 'Close' :  feat_df.Close[lastRow-1]} , ignore_index=True)
    sst.tail(3)
    
    # Update with gainAhead
    end = sst.index[-2]
    sst.iloc[end,sst.columns.get_loc('gainAhead')] = new_st_df.iloc[end,new_st_df.columns.get_loc('gainAhead')]
    sst.tail(3)
    
    # Update TMS Part1 with safe-f, CAR25
    nrows = sst.shape[0]
    # sst = sst.set_index(pd.DatetimeIndex(sst['Date']))
    start = sst.index[0]
    end = sst.index[-1]
    
    iStart = sst.index.get_loc(end)-1
    iEnd = sst.index.get_loc(end)
    
    # retrieve tms_dict
    file_name = 'tms_dict.json'
    tms_dict = dSet.load_json(system_directory, file_name)
    
    forecastHorizon = tms_dict["forecastHorizon"]
    initialEquity = tms_dict["initialEquity"]
    ddTolerance = tms_dict["ddTolerance"]
    tailRiskPct = tms_dict["tailRiskPct"]
    windowLength = tms_dict["windowLength"]
    nCurves = tms_dict["nCurves"]
    updateInterval = tms_dict["updateInterval"]
    
    years_in_forecast = forecastHorizon / 252.0
    
    printDetails = False
    
    # Calculate safe-f, CAR25
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
    
    df_to_save = sst.copy()
    #df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    filename = "TMS_Part1_daybyday_" + system_name + ".csv"
    df_to_save.to_csv(system_directory+ "\\" + filename, encoding='utf-8', index=False)
    
    # Now go to TMS Part 2
    filename = "TMS_Part2_daybyday_" + system_name + ".csv"
    path = system_directory+ "\\" + filename
    tms2 = pd.read_csv(path)
    tms2.tail(3)
    
    # Append last day form TMS Part 1 to TMS Part 2
    tms21 = tms2.copy()
    sst1 = sst.copy()
    #sst1.reset_index(level=sst1.index.names, inplace=True)
    tms21.loc[sst1.index[-1]] = sst1.iloc[-1]
    
    nrows = tms21.shape[0]
    start = tms21.index[0]
    end = tms21.index[-1]
    
    iStart = tms21.index.get_loc(end)-1
    iEnd = tms21.index.get_loc(end)
    
    # Update gainAhead
    tms21.iloc[iStart,tms21.columns.get_loc('gainAhead')] = sst1.iloc[iStart,sst1.columns.get_loc('gainAhead')]
    
    # Update trade_decision with current date decision
    if y_validate[0] == 1: 
        if tms21.iloc[iEnd,tms21.columns.get_loc('CAR25')] > 10:
            tms21.iloc[iEnd,tms21.columns.get_loc('trade_decision')] = 'Long'
        else:
            tms21.iloc[iEnd,tms21.columns.get_loc('trade_decision')] = 'Flat'
    else:
        tms21.iloc[iEnd,tms21.columns.get_loc('trade_decision')] = 'Flat'
        
    for i in range(iStart, iEnd):
        if (tms21.trade_decision[i] == 'Long'):
            tms21.iloc[i,tms21.columns.get_loc('trade')] = tms21.iloc[i-1,tms21.columns.get_loc('fract')] * tms21.iloc[i-1,tms21.columns.get_loc('equity')] * tms21.iloc[i,tms21.columns.get_loc('gainAhead')]
        elif np.logical_and((tms21.signal[i] > 0), (tms21.CAR25[i] > 10)):
            tms21.iloc[i,tms21.columns.get_loc('trade')] = tms21.iloc[i-1,tms21.columns.get_loc('fract')] * tms21.iloc[i-1,tms21.columns.get_loc('equity')] * tms21.iloc[i,tms21.columns.get_loc('gainAhead')]
        else:
            tms21.iloc[i,tms21.columns.get_loc('trade')] = 0.0
            
        tms21.iloc[i,tms21.columns.get_loc('fract')] = tms21.iloc[i,tms21.columns.get_loc('safef')]
        
        tms21.iloc[i,tms21.columns.get_loc('equity')] = tms21.iloc[i-1,tms21.columns.get_loc('equity')] + tms21.iloc[i,tms21.columns.get_loc('trade')]
        tms21.iloc[i,tms21.columns.get_loc('maxEquity')] = max(tms21.iloc[i,tms21.columns.get_loc('equity')],tms21.iloc[i-1,tms21.columns.get_loc('maxEquity')])
        tms21.iloc[i,tms21.columns.get_loc('drawdown')] = (tms21.iloc[i,tms21.columns.get_loc('maxEquity')] - tms21.iloc[i,tms21.columns.get_loc('equity')]) / tms21.iloc[i,tms21.columns.get_loc('maxEquity')]
        tms21.iloc[i,tms21.columns.get_loc('maxDD')] =  max(tms21.iloc[i,tms21.columns.get_loc('drawdown')],tms21.iloc[i-1,tms21.columns.get_loc('maxDD')])
        tms21.iloc[i,tms21.columns.get_loc('fract')] = tms21.iloc[i,tms21.columns.get_loc('safef')]
        
    print(tms21.tail(4))
        
    df_to_save = tms21.copy()
    #df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    filename = "TMS_Part2_daybyday_" + system_name + ".csv"
    df_to_save.to_csv(system_directory+ "\\" + filename, encoding='utf-8', index=False)
    
    system_dict['pivotDate']=tradeDate

    dSet.save_json('system_dict.json', system_directory, system_dict)
    
    plot_tms = tms21.copy()
    plot_tms = plot_tms.set_index(pd.DatetimeIndex(plot_tms['Date']))
    plot_tms=plot_tms.drop('Date', axis=1)
    plotTitle = "Equity curve for  " + issue + ", " + str(start) + " to " + str(end)
    plotIt.plot_v1(plot_tms['equity'][:-2], plotTitle)
    plotTitle = "Drawdown for  " + issue + ", " + str(start) + " to " + str(end)
    plotIt.plot_v1(plot_tms['drawdown'][:-2], plotTitle)
    
    # Plot code
    fig = plt.figure(figsize=(11,6))
    fig.suptitle('CAR25 and issue price for' + issue)
    ax1 = fig.add_subplot(111)
    #ax1.plot(sst1.safef, color='green',label='safe-f')
    ax1.plot(plot_tms.CAR25, color='blue',label='CAR25')
    #ax1.plot(valData.equityValBeLongSignals, color='purple',label='ValBeLong')
    
    ax1.legend(loc='upper left', frameon=True, fontsize=8)
    ax1.label_outer()
    ax1.tick_params(axis='x',which='major',bottom=True)
    ax1.minorticks_on()
    ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
    ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
    
    #sst1['Pri']=valData.Pri
    ax2 = ax1.twinx()
    ax2.plot(plot_tms.Close, color='black',alpha=0.6,label='CLOSE',linestyle='--')
    ax2.legend(loc='center left', frameon=True, fontsize=8)
    ax2.label_outer()
    fig.autofmt_xdate()
    
    plotTitle = "Safe-f for " + issue
    plotIt.plot_v1(plot_tms['safef'][:-2], plotTitle)
