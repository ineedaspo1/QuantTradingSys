# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:35:32 2019

@author: KRUEGKJ
IS day by day
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
import sys

#from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Import custom libraries
from Code.lib.plot_utils import PlotUtility
from Code.lib.time_utils import TimeUtility
from Code.lib.retrieve_data import DataRetrieve, ComputeTarget
from Code.lib.retrieve_system_info import TradingSystemUtility
#from Code.lib.candle_indicators import CandleIndicators
from Code.lib.transformers import Transformers
#from Code.lib.ta_momentum_studies import TALibMomentumStudies
from Code.lib.model_utils import ModelUtility, TimeSeriesSplitImproved
from Code.lib.feature_generator import FeatureGenerator
#from Code.utilities.stat_tests import stationarity_tests
from Code.lib.config import current_feature, feature_dict
from Code.models import models_utils
from Code.lib.model_algos import AlgoUtility

plotIt = PlotUtility()
timeUtil = TimeUtility()
ct = ComputeTarget()
#candle_ind = CandleIndicators()
dSet = DataRetrieve()
#taLibMomSt = TALibMomentumStudies()
transf = Transformers()
modelUtil = ModelUtility()
featureGen = FeatureGenerator()
modelAlgo = AlgoUtility()
sysUtil = TradingSystemUtility()

if __name__ == '__main__':

    # set to existing system name OR set to blank if creating new
    if len(sys.argv) < 2:
        print('You must set a system_name or set to """"!!!')
    
    system_name = sys.argv[1]
    ext_input_dict = sys.argv[2]
    
    # This should be a function . . . 
    # pass system name, do the rest, return system_dict
    if system_name == "":
        # set some defaults for now 
        print("New system")
        issue = "TLT"
        direction = "Long"
        ver_num = 3
        system_dict = sysUtil.get_system_dict(system_name, issue, direction, ver_num)
    
        pivotDate = str(datetime.date(2019, 1, 3)) # need as string for serialization
        is_oos_ratio = 2
        oos_months = 6
        segments = 2
    
        system_dict['pivotDate'] = pivotDate
        system_dict['is_oos_ratio'] = is_oos_ratio
        system_dict['oos_months'] = oos_months
        system_dict['segments'] = segments
    
        system_name = system_dict['system_name']
        sysUtil.save_dict(system_name, 'system_dict', system_dict)
    else:
        print("Existing system")
        system_directory = sysUtil.get_system_dir(system_name)
        system_dict = sysUtil.get_dict(system_directory, 'system_dict')
        is_oos_ratio = system_dict["is_oos_ratio"]
        oos_months = system_dict["oos_months"]
        segments = system_dict["segments"]
        issue = system_dict['issue']
        
    print(system_dict)
    system_directory = sysUtil.get_system_dir(system_name)  
    # Set IS-OOS parameters
    pivotDate = system_dict['pivotDate']
    
    #pivotDate = datetime.strptime(pivotDate, '%Y-%m-%d')
    print(pivotDate)
    
    # Read full dataset for issue
    df = dSet.read_issue_data(issue)

    # Set date range and target
    dataLoadStartDate = df.Date[0]
    lastRow = df.shape[0]
    dataLoadEndDate = df.Date[lastRow-1]
    dataSet = dSet.set_date_range(df, dataLoadStartDate,dataLoadEndDate)
    # Resolve any NA's for now
    dataSet.fillna(method='ffill', inplace=True)
    
    # set beLong level
    # that should be saved in system_dict
    beLongThreshold = 0.000
    dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)
    
    print(dataSet.tail(3))
    
    # Create features
    # 
    # Check for external dict file
    if ext_input_dict == "":
        print("no external input_dict file, reference coded parameters")    
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
        
    else:
        # If new system, separately generated input_dict will not be in
        # system_directory...
        # assume input_dicts local to code
        current_directory = os.getcwd()
        # add try/catch block later
        filename = ext_input_dict
        input_dict = dSet.load_pickle(current_directory, filename)
        print(input_dict)
    
    # save ext input name to system_dict
    system_dict['extInputDict'] = ext_input_dict
    sysUtil.save_dict(system_name, 'system_dict', system_dict)
    
    # now save locally to system
    sysUtil.save_dict(system_name, 'input_dict', input_dict)    

    dataSet2 = featureGen.generate_features(dataSet, input_dict)
    current_feature['Latest'] = 'Volume'
    dataSet2 = transf.normalizer(dataSet, 'Volume', 50)
    print(dataSet2.tail(5))
    
    # save Dataset of analysis
    # THIS SHOULD BE A FUNCTION
    print("====Saving dataSet====\n")
    file_title = "raw-features-" + system_name + ".pkl"
    file_name = os.path.join(system_directory, file_title)
    dataSet2.to_pickle(file_name)
    
    # Examine correlations of features
    # Get columns to drop from feature_dict
    col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
    # And set OHLC, etc., to Drop for cleaner correlation analysis
    to_drop = ['Open','High','Low', 'gainAhead', 'Close', 'beLong', 'AdjClose']
    for x in to_drop:
        col_vals.append(x)
    mmData = dSet.drop_columns(dataSet2, col_vals)
    plotIt.correlation_matrix(mmData)
    
    # Examine and drop feature with corr value > 0.85
    # Create correlation matrix
    corr_matrix = mmData.corr()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.85
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    print('Column(s) to drop: %s' % to_drop)
    
    # If there are columns to Drop, change feature dict to indicate Drop
    if len(to_drop) > 0:
        for x in to_drop:
            feature_dict[x] = 'Drop'
        print(feature_dict)
    # Save feature_dict
    sysUtil.save_dict(system_name, 'feature_dict', feature_dict) 

    #
    # Starting to set time frames for IS analysis
    # Make sure analysis is complete for all time segments and
    # decide how to use models
    # set date splits
    new_pivot_date = datetime.datetime.strptime(pivotDate, '%Y-%m-%d').date()
    isOosDates = timeUtil.is_oos_data_split(issue, new_pivot_date, is_oos_ratio, oos_months, segments)
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
    
    # Prep dataset for classification
    model_results = []
    
    # Start IS
    for i in range(segments):        
        mmData = dataSet[modelStartDate:modelEndDate].copy()
        # EV related
        evData = dataSet2.loc[modelStartDate:modelEndDate].copy()
        nrows = mmData.shape[0]
        
        plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
        plotIt.plot_v2x(mmData, plotTitle)
        plotIt.histogram(mmData['beLong'],
                         x_label="beLong signal",
                         y_label="Frequency",
                         title = "beLong distribution for " + issue
                         )        
        plt.show(block=False)
            
        col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
        to_drop = ['Open','High','Low', 'gainAhead', 'Close', 'Volume', 'AdjClose']
        for x in to_drop:
            col_vals.append(x)
            
        mmData = dSet.drop_columns(mmData, col_vals)
        nrows = mmData.shape[0]
        print("rows of mmData", nrows)
        
        # Prep for classification
        ######################
        # ML section
        ######################
        #  Make 'iterations' index vectors for the train-test split
        iterations = 100
        #tscv = TimeSeriesSplit(n_splits=4)
        
        dX, dy = modelUtil.prepare_for_classification(mmData)        
        
        tscvi = TimeSeriesSplitImproved(n_splits=5)
        
        # Make predictions with models
        to_model = {"RF": modelAlgo.setRFModel(min_samples_split=20,
                                               n_estimators=200,
                                               max_features=None
                                               ),
                    "KNN": modelAlgo.setKNNModel(n_neighbors=3),
                    "SVM": modelAlgo.setSVMModel(),
                    "GTB": modelAlgo.setGTBModel(learning_rate=0.1,
                                                 subsample=0.5,
                                                 n_estimators=20,
                                                 min_samples_leaf=20,
                                                 max_depth=3,
                                                 max_features='auto',
                                                 random_state = 0
                                                ),
                    "QDA": modelAlgo.setQDAModel(),
                    "LogReg": modelAlgo.setLogRegModel(solver='lbfgs',
                                                       max_iter=1000)}
        for key, value in to_model.items():
            modelname = key
            model = value
            info_dict = {'issue':issue,
                         'modelStartDate':modelStartDate,
                         'modelEndDate':modelEndDate,
                         'modelname':modelname,
                         'nrows':nrows,
                         'system_name':system_name
                        }
            print(modelname)
            #print(model)
        
            model_results, fit_model = modelUtil.model_and_test(dX,
                                                                dy,
                                                                model,
                                                                model_results,
                                                                tscvi,
                                                                info_dict,
                                                                evData
                                                               )
            
            # save Dataset of analysis
            print("====Saving model====\n")
            file_title = "fit-model-"+modelname+"-IS-"+system_name+"-segment-"+str(i)+".sav"
            file_name = os.path.join(system_directory, file_title)
            pickle.dump(fit_model, open(file_name, 'wb'))
            #print(model_results)
            
        modelStartDate = modelStartDate  + relativedelta(months=oos_months) + BDay(1)
        #print(modelStartDate)
        modelEndDate = modelStartDate + relativedelta(months=is_months) - BDay(1)
        #print(modelEndDate)
        
    ## loop ended, print results
    df = pd.DataFrame(model_results)
    df = df[['Issue',
             'StartDate',
             'EndDate',
             'Model',
             'Rows',
             'beLongCount',
             'Features',
             'FeatureCount',
             'Train-Accuracy',
             'Train-Precision',
             'Train-RMC',
             'Train-RF',
             'Train-NPV',
             'Train-MCC',
             'Train-EV',
             'Test-Accuracy',
             'Test-Precision',
             'Test-RMC',
             'Test-RF',
             'Test-NPV',
             'Test-MCC',
             'Test-EV',
            ]]
    #print(df)
    
    # Save results
    dSet.save_csv(system_directory,
                  system_name,
                  'IS_Equity',
                  'new', 
                  df
                  )
    