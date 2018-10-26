# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:19:51 2018

@author: kruegkj

insample_sample_model.py
"""
import sys
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import PlotUtility
from time_utils import *
from retrieve_data import *
from candle_indicators import *
from transformers import *
from ta_momentum_studies import *
from model_utils import *
from feature_generator import *
from config import current_feature, feature_dict

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
import os.path
import pickle
from sklearn.externals import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import BDay

us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

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
    is_oos_ratio = 4
    oos_months = 3
    segments = 3
    
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
        to_drop = ['Open','High','Low', 'gainAhead', 'Symbol', 'Date']
        for x in to_drop:
            col_vals.append(x)
        mmData = dSet.drop_columns(mmData, col_vals)
        
        ### save column names to file

        ######################
        # ML section
        ######################
        #  Make 'iterations' index vectors for the train-test split
        iterations = 50
        info_dict = {'issue':issue, 'modelStartDate':modelStartDate, 'modelEndDate':modelEndDate, 'modelname':modelname, 'nrows':nrows, 'predictors':predictor_vars}     
        
        dX, dy = modelUtil.prepare_for_classification(mmData)        
        
        sss = StratifiedShuffleSplit(n_splits=iterations,test_size=0.33, random_state=None)
                
        model = RandomForestClassifier(n_jobs=-1, random_state=55, min_samples_split=10 , n_estimators=500, max_features = 'auto', min_samples_leaf = 10, oob_score = 'TRUE')
                
        ### add issue, other data OR use dictionary to pass data!!!!!!
        model_results = modelUtil.model_and_test(dX, dy, model, model_results, sss, info_dict)
        
        modelStartDate = modelEndDate  + BDay(1)
        modelEndDate = modelStartDate + relativedelta(months=is_months) - BDay(1)
 
    df = pd.DataFrame(model_results)
    df = df[['Issue','StartDate','EndDate','Model','Rows','beLongCount','Predictors','IS-Accuracy','IS-Precision','IS-Recall','IS-F1','OOS-Accuracy','OOS-Precision','OOS-Recall','OOS-F1']]
    print(df)

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
