# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:34:25 2019

@author: kruegkj
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
import sys

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
from Code.lib.config import current_feature, feature_dict
from Code.models import models_utils
from Code.lib.model_algos import AlgoUtility
from Code.lib.tms_utils import TradeRisk

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
tradeRisk = TradeRisk()


def get_dict(system_name, dict_name):
    dict_lookup = {'system_dict': 'system_dict.json',
                   'feature_dict': 'feature_dict.json',
                   'input_dict': 'input_dict.pkl',
                   'tms_dict': 'feature_dict.json'}
    
    system_directory = sysUtil.get_system_dir(system_name)
    if not os.path.exists(system_directory):
        print("system doesn't exist")
    else:
        file_name = dict_lookup[dict_name]
        fn_split = file_name.split(".",1)
        file_suffix = fn_split[1]
        if file_suffix == 'json':
            return_dict = dSet.load_json(system_directory, file_name)
        elif file_suffix == 'pkl':
            return_dict = dSet.load_pickle(system_directory, file_name)
        else:
            print('dict type not found')
            sys.exit()
    return return_dict

def save_dict(system_name, dict_name, dict_file):
    dict_lookup = {'system_dict': 'system_dict.json',
                   'feature_dict': 'feature_dict.json',
                   'input_dict': 'input_dict.pkl',
                   'tms_dict': 'feature_dict.json'}
    
    system_directory = sysUtil.get_system_dir(system_name)
    if not os.path.exists(system_directory):
        print("system doesn't exist")
    else:
        file_name = dict_lookup[dict_name]
        fn_split = file_name.split(".",1)
        file_suffix = fn_split[1]
        if file_suffix == 'json':
            dSet.save_json(file_name, system_directory, dict_file)
            print(file_name + ' saved.')
        elif file_suffix == 'pkl':
            dSet.save_pickle(file_name, system_directory, dict_file)
            print(file_name + ' saved.')
        else:
            print('dict type not found')
            sys.exit()        

if __name__ == '__main__':
    
    # set to existing system name OR set to blank if creating new
    if len(sys.argv) < 2:
        print('You must set a system_name or set to """"!!!')
    
    system_name = sys.argv[1]
    system_directory = sysUtil.get_system_dir(system_name)
    #ext_input_dict = sys.argv[2]
    
    print("Existing system")
    
    # Get info from system_dict
    system_dict = get_dict(system_name, 'system_dict')
    issue = system_dict["issue"]
    
    is_oos_ratio = system_dict["is_oos_ratio"]
    oos_months = system_dict["oos_months"]
    segments = system_dict["segments"]
    
    print(system_dict)
    
    # Set IS-OOS parameters
    pivotDate = system_dict["pivotDate"]
    pivotDate = datetime.strptime(pivotDate, '%Y-%m-%d')
    print('pivotDate: ', pivotDate)
    pivotDate = datetime.date(pivotDate)
    
    # set date splits
    # TO DO: move creation of and reading of params into dict
    isOosDates = timeUtil.is_oos_data_split(issue, pivotDate, is_oos_ratio, oos_months, segments)
    dataLoadStartDate = isOosDates[0]
    is_start_date = isOosDates[1]
    oos_start_date = isOosDates[2]
    is_months = isOosDates[3]
    is_end_date = isOosDates[4]
    oos_end_date = isOosDates[5]
    
    modelStartDate = oos_start_date
    
    # Retrieve feature_dict
    feature_dict = get_dict(system_name, 'feature_dict')
    
    # get raw data 
    raw_df = dSet.read_issue_data(issue)
    
    temp_raw_df = raw_df.copy()
    
    # Set trade date
    from datetime import timedelta
    tradeDate = pivotDate + BDay(1)
    print('tradeDate: ', tradeDate)

    if (tradeDate.strftime('%Y-%m-%d') > raw_df['Date'].values.all()):
        print ('trade date > last date in raw_df')
        skip_input = input("Skip date?: (Y/N)")
        if skip_input == 'Y':
            system_dict['pivotDate']=tradeDate.strftime('%Y-%m-%d')
            save_dict(system_name, 'system_dict', system_dict)
            sys.exit()  
        open_input = input("Enter Open:")
        high_input = input("Enter High:")
        low_input = input("Enter Low:")
        close_input = input("Enter Close:")
        adj_close_input = input("Enter Adjusted Close:")
        volume_input = input("Enter Volume:")
        
        temp_raw_df = raw_df.append({'Date' : tradeDate.strftime('%Y-%m-%d'), 'Open' : float(open_input), 'High' : float(high_input), 'Low' : float(low_input), 'Close' : float(close_input), 'AdjClose' : float(adj_close_input), 'Volume' : float(volume_input) } , ignore_index=True)
        
        print(temp_raw_df.tail(3))
        raw_df = temp_raw_df.copy()
        print(raw_df.tail(3))
        
        dSet.save_pickle_price_data(issue, raw_df)

    else:
        print('no date')
        tradeDate += BDay(1)
        print('tradeDate: ', tradeDate)
        
    # set trade date
    tradeDate = tradeDate.strftime('%Y-%m-%d')
    print(tradeDate)
    
#    # set df to trade date range
#    trimmed_df = raw_df.iloc[0:raw_df[raw_df.Date == tradeDate].index[0]]
#    print(trimmed_df.tail(2))
#
#    price_loc = raw_df.index[raw_df.Date == tradeDate]
#    index = price_loc[0]
#    print(index)
#
#    # get Close from trade date
#    new_open = raw_df.Close[index]
#    print(new_open)
#
#    # append price to df
#    new_data_df = trimmed_df.append({'Date' : tradeDate , 'Open' : raw_df.Open[index], 'High' : raw_df.High[index], 'Low' : raw_df.Low[index], 'Close' : raw_df.Close[index], 'AdjClose' : raw_df.AdjClose[index], 'Volume' : raw_df.Volume[index] } , ignore_index=True)
#    print(new_data_df.tail(2))
#
    # get first row
    df_start_date = temp_raw_df.Date[0]
    
    # get last row
    lastRow = temp_raw_df.shape[0]
    df_end_date = temp_raw_df.Date[lastRow-1]
    
    feat_df = dSet.set_date_range(temp_raw_df, df_start_date, df_end_date)
    # Resolve any NA's for now
    feat_df.fillna(method='ffill', inplace=True)
    
    #set beLong level
    beLongThreshold = 0.000
    feat_df = ct.setTarget(temp_raw_df, "Long", beLongThreshold)
    
    print(feat_df.tail(2))

    # Adding features with new day
    input_dict = get_dict(system_name, 'input_dict')
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
    print(y_validate[0])
    
    filename = "OOS_Equity_daybyday_" + system_name + ".csv"
    path = system_directory+ "\\" + filename
    shadow_trades = pd.read_csv(path)
    
    # Update shadow trades
    new_st_df = shadow_trades.append({'Date' : tradeDate , 'signal' : y_validate[0], 'gainAhead' : 0.000, 'Close' : feat_df.Close[lastRow-1] } , ignore_index=True)
    
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
    
    end = sst.index[-1]
    iStart = sst.index.get_loc(end)-1
    iEnd = sst.index.get_loc(end)
    
    # retrieve tms_dict
    file_name = 'tms_dict.json'
    tms_dict = dSet.load_json(system_directory, file_name)
    
    sst = tradeRisk.get_safef_car25(sst, iStart, iEnd, tms_dict)
    
    #df_to_save = sst.copy()
    #df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    filename = "TMS_Part1_daybyday_" + system_name + ".csv"
    sst.to_csv(system_directory+ "\\" + filename, encoding='utf-8', index=False)
    
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
    
    tms21 = tradeRisk.update_tms(tms21, iStart, iEnd, y_validate)
    
    print(tms21.tail(4))
        
    df_to_save = tms21.copy()
    #df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
    filename = "TMS_Part2_daybyday_" + system_name + ".csv"
    df_to_save.to_csv(system_directory+ "\\" + filename, encoding='utf-8', index=False)
    
    system_dict['pivotDate']=tradeDate
    save_dict(system_name, 'system_dict', system_dict)
    
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
