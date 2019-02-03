# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018

@author: KRUEGKJ

model_utils.py
"""
from Code.lib.config import current_feature, feature_dict
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class AlgoUtility:
    def setRFClass(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = RandomForestClassifier(n_jobs=-1,
                                           random_state=55,
                                           oob_score = 'TRUE',
                                           **parameters
                                           )
        print(model)
        return model
    
    def setKNNClass(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = KNeighborsClassifier(**parameters)
        print(model)
        return model

    def setSVMClass(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = svm.SVC(shrinking=False,
                            random_state=0,
                            gamma='scale',
                            **parameters
                            )    
        print(model)
        return model
    
    def setAdaBoostClass(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=10),
                                       **parameters
                                       )
        print(model)
        return model
    
    def setGTBClass(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = GradientBoostingClassifier(**parameters)
        print(model)
        return model
    
    def setQDAClass(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = QuadraticDiscriminantAnalysis(**parameters)
        print(model)
        return model
    
    
    


if __name__ == "__main__":
    from Code.lib.plot_utils import PlotUtility
    from Code.lib.retrieve_data import DataRetrieve, ComputeTarget
    from Code.lib.time_utils import TimeUtility
    from Code.lib.feature_generator import FeatureGenerator
    from Code.utilities.stat_tests import stationarity_tests
    from Code.lib.model_utils import ModelUtility, TimeSeriesSplitImproved
#    from Code.lib.config import current_feature, feature_dict
    
    plotIt = PlotUtility()
    ct = ComputeTarget()
    dSet = DataRetrieve()
    featureGen = FeatureGenerator()
    timeUtil = TimeUtility()
    modelUtil = ModelUtility()
    modelAlgo = AlgoUtility()
    #timeSeriesImp = TimeSeriesSplitImproved()
    
    dataLoadStartDate = "2014-09-26"
    dataLoadEndDate = "2018-04-05"
    issue = "TLT"
    # Set IS-OOS parameters
    pivotDate = datetime.date(2018, 4, 2)
    is_oos_ratio = 2
    oos_months = 4
    segments = 4
    
    dataSet = dSet.read_issue_data(issue)
    
    dataSet = dSet.set_date_range(dataSet,
                                  dataLoadStartDate,
                                  pivotDate
                                  )
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
    dataSet = featureGen.generate_features(dataSet, input_dict)
    
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
    
    mmData = dataSet.loc[modelStartDate:modelEndDate].copy()
    # EV related
    evData = dataSet.loc[modelStartDate:modelEndDate].copy()
    
    col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
    to_drop = ['Open','High','Low', 'gainAhead', 'Symbol', 'Date', 'Close']
    for x in to_drop:
        col_vals.append(x)
    mmData = dSet.drop_columns(mmData, col_vals)
    nrows = mmData.shape[0]

    ######################
    # ML section
    ######################
    #  Make 'iterations' index vectors for the train-test split
    iterations = 100
    tscv = TimeSeriesSplit(n_splits=10)
    
    dX, dy = modelUtil.prepare_for_classification(mmData)        
    
    sss = StratifiedShuffleSplit(n_splits=iterations,
                                 test_size=0.33,
                                 random_state=None
                                 )
    
    tscv = TimeSeriesSplit(n_splits=6, max_train_size=24)
    
    predictor_vars = "convert info_dict to columns to insert"
    model_results = []
 
    to_model = {"RF": modelAlgo.setRFClass(min_samples_split=20,
                                           n_estimators=200,
                                           max_features=None
                                           ),
                "KNN": modelAlgo.setKNNClass(n_neighbors=5)}
    for key, value in to_model.items():
        modelname = key
        model = value
        info_dict = {'issue':issue, 'modelStartDate':modelStartDate, 'modelEndDate':modelEndDate, 'modelname':modelname, 'nrows':nrows}
        print(modelname)
        print(model)
        
        model_results = modelUtil.model_and_test(dX, dy, model, model_results, sss, info_dict, evData)
    print(model_results)
