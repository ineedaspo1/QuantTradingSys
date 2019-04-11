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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from statsmodels.stats.outliers_influence import variance_inflation_factor    

class AlgoUtility:
    # RF
    # KNN
    # SVM
    # AdaBoost
    # GTB
    # QDA
    # LogReg
    # Gaussian Process
    
    def setExtraTreesClassifier(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = ExtraTreesClassifier(**parameters)
        print(model)
        return model
    
    def setGaussianNBModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = GaussianNB(**parameters)
        print(model)
        return model
    
    def setMLPClassifierModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = MLPClassifier(**parameters)
        print(model)
        return model
    
    def setDecisionTreeModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = DecisionTreeClassifier(**parameters)
        print(model)
        return model
    
    def setGausianProcessModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = GaussianProcessClassifier(1.0 * RBF(1.0),
                                          warm_start=True,
                                          **parameters
                                           )
        print(model)
        return model
    
    def setRFModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = RandomForestClassifier(n_jobs=-1,
                                           random_state=55,
                                           oob_score = 'TRUE',
                                           **parameters
                                           )
        print(model)
        return model
    
    def setKNNModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = KNeighborsClassifier(**parameters)
        print(model)
        return model

    def setSVMModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = svm.SVC(shrinking=False,
                            random_state=0,
                            gamma='scale',
                            **parameters
                            )    
        print(model)
        return model
    
    def setAdaBoostModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=10),
                                       **parameters
                                       )
        print(model)
        return model
    
    def setGTBModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = GradientBoostingClassifier(**parameters)
        print(model)
        return model
    
    def setQDAModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = QuadraticDiscriminantAnalysis(**parameters)
        print(model)
        return model
    
    def setLDAModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = LinearDiscriminantAnalysis(**parameters)
        print(model)
        return model
    
    def setLogRegModel(self, **parameters):
        if parameters is not None:
            params_used = parameters
        model = LogisticRegression(**parameters)
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
    
    def calculate_vif_(X, thresh=5.0):
        variables = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                   for ix in range(X.iloc[:, variables].shape[1])]
    
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True
    
        print('Remaining variables:')
        print(X.columns[variables])
        return X.iloc[:, variables]
    
    dataLoadStartDate = "2014-09-26"
    dataLoadEndDate = "2019-04-05"
    issue = "TLT"
    # Set IS-OOS parameters
    pivotDate = datetime.date(2019, 1, 3)
    is_oos_ratio = 2
    oos_months = 8
    segments = 1
    
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
                      {'fname' : 'RSI', 
                       'params' : [8],
                       'transform' : ['Normalized', 20]
                       },
                      'f2': 
                      {'fname' : 'Lag', 
                       'params' : [7],
                       'transform' : ['Normalized', 20]
                       },
                      'f3': 
                      {'fname' : 'MFI', 
                       'params' : [3],
                       'transform' : ['Scaler', 'Robust']
                       },
                      'f4': 
                      {'fname' : 'DetrendPO', 
                       'params' : [6],
                       'transform' : ['Normalized', 20]
                       },
                      'f5': 
                      {'fname' : 'NATR', 
                       'params' : [9],
                       'transform' : ['Normalized', 50]
                       },
                      'f6': 
                      {'fname' : 'exp_MA', 
                       'params' : [4],
                       'transform' : ['Normalized', 20]
                       },
                      'f7': 
                      {'fname' : 'triangMA', 
                       'params' : [4],
                       'transform' : ['Normalized', 20]
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
    to_drop = ['Open','High','Low', 'gainAhead', 'Close', 'AdjClose', 'Volume']
    for x in to_drop:
        col_vals.append(x)
    mmData = dSet.drop_columns(mmData, col_vals)
    nrows = mmData.shape[0]
    
    X = mmData.copy()
    
    
    XX = calculate_vif_(X)

    ######################
    # ML section
    ######################
    #  Make 'iterations' index vectors for the train-test split
    iterations = 100
    #tscv = TimeSeriesSplit(n_splits=10)
    
    dX, dy = modelUtil.prepare_for_classification(mmData)        
    
#    sss = StratifiedShuffleSplit(n_splits=iterations,
#                                 test_size=0.33,
#                                 random_state=None
#                                 )
    
    tscvi = TimeSeriesSplitImproved(n_splits=4)
    
    predictor_vars = "convert info_dict to columns to insert"
    model_results = []
 
    to_model = {"RF":      modelAlgo.setRFModel(min_samples_split=20,
                                                n_estimators=200,
                                                max_features=None
                                                ),
                "KNN":      modelAlgo.setKNNModel(n_neighbors=5),
                "LogReg":   modelAlgo.setLogRegModel(solver='lbfgs',
                                                     max_iter=1000),
                "QDA":      modelAlgo.setQDAModel(),
                "LDA":      modelAlgo.setLDAModel(),
                "GP":       modelAlgo.setGausianProcessModel(),
                "DecTree":  modelAlgo.setDecisionTreeModel(max_depth=5),
                "MLP":      modelAlgo.setMLPClassifierModel(max_iter=1000,
                                                            learning_rate_init=0.01
                                                            ),
                "GNB":      modelAlgo.setGaussianNBModel(),
                "GTB":      modelAlgo.setGTBModel(),
                "ETC":      modelAlgo.setExtraTreesClassifier(),
                "AdaBoost": modelAlgo.setAdaBoostModel(),
                "SVM":      modelAlgo.setSVMModel()
                }
    
    for key, value in to_model.items():
        modelname = key
        model = value
        info_dict = {'issue':issue, 'modelStartDate':modelStartDate, 'modelEndDate':modelEndDate, 'modelname':modelname, 'nrows':nrows}
        print(modelname)
        print(model)
        
        model_results, fit_model = modelUtil.model_and_test(dX, dy, model, model_results, tscvi, info_dict, evData)
    print(model_results)
