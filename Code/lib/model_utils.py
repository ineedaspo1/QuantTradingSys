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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class ModelUtility:
    
    def conf_matrix_results(self, cm_results, printResults=False):
        ''' Function to define CM results and details of other measures
            Accuracy (also Rate of Correctness): Percentage of accurate predictions
            Precision: Positive predicted value(PPV)
            Rate of Missing Chances (RMC): FN / # of true instances
            Rate of Failure (RF): FP / # positive instances
            Negative Predicted Value (NPV)
            MCC: In essence a correlation coefficient between the observed 
            and predicted binary classifications. It returns a value between
            −1 and +1. A coefficient of +1 represents a perfect prediction. 
            0 no better than random prediction. −1 indicates total disagreement 
            between prediction and observation.
            '''
        return_cm = {}
        
        tp = cm_results[1,1]
        fn = cm_results[1,0]
        fp = cm_results[0,1]
        tn = cm_results[0,0]
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        accuracy = (tp + tn)/(tp + fn + fp + tn)
        specificity = tn/(tp+tn)
        rmc = fn/(fn+tp)
        rf = fp/(fp + tp)
        npv = tn/(tn+fn)
        f1 = (2.0 * precision * recall) / (precision + recall)
        mcc = ((tp*tn) - (fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

        return_cm['accuracy'] = accuracy
        return_cm['precision'] = precision
        return_cm['rmc'] = rmc
        return_cm['rf'] = rf
        return_cm['npv'] = npv
        return_cm['mcc'] = mcc
        
        if printResults:
            print('{0:>30} {1:2.2f}'.format("Accuracy: ", accuracy))
            print('{0:>30} {1:2.2f} {2}'.format("Recall: ", recall, "(When 1, how often prediction correct?)"))
            print('{0:>30} {1:2.2f} {2}'.format("Specificity: ", specificity, "(When -1, how often prediction correct?)"))
            print('{0:>30} {1:2.2f}'.format("Precision (PPV): ", precision))
            print('{0:>30} {1:2.2f}'.format("Rate of Missing Chances: ", rmc))
            print('{0:>30} {1:2.2f}'.format("Rate of Failure: ", rf))
            print('{0:>30} {1:2.2f}'.format("Neg Pred Value (NPV): ", npv))
            print('{0:>30} {1:2.2f}'.format("MCC: ", mcc))
        return return_cm
 
    def cm_plot(self, cm, type):
        '''Function for Confusion Matrix Plot
            Print CM for totals and percentages
        '''
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title(type + ' beLong Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                # counts
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j])+"\n")
                # percentage of total
                plt.text(j,i, str('{:.2%}'.format(cm[i][j]/cm.sum())))
        plt.show()
        
    def get_evData_avg(self, evData, signal):
        '''Get average gain/loss from beLong signals
           Across data set, sum gainAhead and count number signals to 
           provide an average gain (mutiply * 100 to get %)
        '''
        # sum gainAhead of signals
        ev_sum = evData.loc[evData['beLong'] == signal, 'gainAhead'].sum()
        ev_cnt = evData.loc[evData['beLong'] == signal, 'beLong'].count()
        ev_avg = ev_sum/ev_cnt
        return ev_avg
    
    def find_expected_value(self, cm, cost_benefit): 
        ''' Expected Value
            Associate the confusion matrix with an expected value using a cost 
            benefit analysis matrix.
            1.Get probabilities for the confusion matrix
            2.Retrieve the average returns for the confusion matrix. In this 
            case, the only returns that affect the expected value are the
                •Average gains for TP
                •Average losses for FP
                •0 loss used for TN, and FN (FN is a missed opportunity, but no loss)
            3.Multiply and CB and CM probabilities
        '''
        # if you use a probability matrix instead, this next line will return the same matrix back
        #print("CB\n", cost_benefit)
        probabilities = cm / cm.sum()
        #print("Probabilities:\n", probabilities)
        ev = probabilities * cost_benefit
        #print("EV array:\n", ev)
        return ev.sum()
    
    def print_expected_value(self, cm_data, cm_type, ev_data):
        print ("----------------------")
        print ("==" + cm_type + "==")
        self.cm_plot(cm_data, cm_type)
        self.conf_matrix_results(cm_data,printResults=True)
        # calculate CB from predicted values
        # populate 2x2 matrix for multiplication with CM values
        ev_not_beLong = self.get_evData_avg(ev_data, -1)
        ev_beLong = self.get_evData_avg(ev_data, 1)
        cb = np.array([[0.0, ev_not_beLong], [0, ev_beLong]])
        exp_value = self.find_expected_value(cm_data, cb)
        print('{0:>30} {1:.3%}'.format("Expected Value: ", exp_value))
        print ("\n")
        return exp_value
    
    def prepare_for_classification(self, df):
        datay = df['beLong']
        dataX = df.drop(['beLong'],axis=1)
        #  Copy from pandas dataframe to numpy arrays
        dy = np.zeros_like(datay)
        dX = np.zeros_like(dataX)
        dy = datay.values
        dX = dataX.values
        return dX, dy
    
    def model_and_test(self, dX, dy, model, model_results, tscv, info_dict, evData):
        accuracy_scores_is = []
        accuracy_scores_oos = []
        precision_scores_is = []
        precision_scores_oos = []
        recall_scores_is = []
        recall_scores_oos = []
        f1_scores_is = []
        f1_scores_oos = []
        hit_rate_is = []
        hit_rate_oos = []
        
        #  Initialize the confusion matrix
        cm_sum_is = np.zeros((2,2))
        cm_sum_oos = np.zeros((2,2))
        #  For each entry in the set of splits, fit and predict

        for train_index,test_index in tscv.split(dX,dy):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = dX[train_index], dX[test_index]
            y_train, y_test = dy[train_index], dy[test_index] 
        #   print("TRAIN:", train_index, "TEST:", test_index)
        #   fit the model to the in-sample data
            model.fit(X_train, y_train)
            
        #  test the in-sample fit    
            y_pred_is = model.predict(X_train)
            #print("%s: %0.3f" % ("Hit rate (IS)  ", model.score(X_train, y_train)))
            cm_is = confusion_matrix(y_train, y_pred_is)
            cm_sum_is = cm_sum_is + cm_is
            accuracy_scores_is.append(accuracy_score(y_train, y_pred_is))
            precision_scores_is.append(precision_score(y_train, y_pred_is))
            recall_scores_is.append(recall_score(y_train, y_pred_is))
            f1_scores_is.append(f1_score(y_train, y_pred_is))
            
        #  test the out-of-sample data
            y_pred_oos = model.predict(X_test)
            #print("%s: %0.3f" % ("Hit rate (OOS) ", model.score(X_test, y_test)))
            cm_oos = confusion_matrix(y_test, y_pred_oos)
            #print(model.score(X_test, y_test))
            cm_sum_oos = cm_sum_oos + cm_oos
            accuracy_scores_oos.append(accuracy_score(y_test, y_pred_oos))
            precision_scores_oos.append(precision_score(y_test, y_pred_oos))
            recall_scores_oos.append(recall_score(y_test, y_pred_oos))
            f1_scores_oos.append(f1_score(y_test, y_pred_oos))

        is_ev = self.print_expected_value(cm_sum_is, "In Sample", evData)
        oos_ev = self.print_expected_value(cm_sum_oos, "Out of Sample", evData)
        
        is_cm_results = self.conf_matrix_results(cm_sum_is, printResults=False)
        #print(is_cm_results)
        oos_cm_results = self.conf_matrix_results(cm_sum_oos, printResults=False)
        #print(oos_cm_results)
        
        col_save = [k for k,v in feature_dict.items() if v == 'Keep']
        
        model_results.append({'Issue': info_dict['issue'],
                              'StartDate': info_dict['modelStartDate'].strftime("%Y-%m-%d"),
                              'EndDate': info_dict['modelEndDate'].strftime("%Y-%m-%d"), 
                              'Model': info_dict['modelname'],
                              'Rows': info_dict['nrows'], 
                              'beLongCount': str(np.count_nonzero(dy==1)), 
                              'Features': col_save, 
                              'IS-Accuracy': np.mean(accuracy_scores_is),
                              'IS-Precision': np.mean(precision_scores_is),
                              'IS-RMC': is_cm_results["rmc"],
                              'IS-RF': is_cm_results["rf"],
                              'IS-NPV': is_cm_results["npv"],
                              'IS-MCC': is_cm_results["mcc"],
                              'IS-Recall': np.mean(recall_scores_is),
                              'IS-F1': np.mean(f1_scores_is),
                              'IS-EV': is_ev*100,
                              'OOS-Accuracy':  np.mean(accuracy_scores_oos),
                              'OOS-Precision': np.mean(precision_scores_oos),
                              'OOS-RMC': oos_cm_results["rmc"],
                              'OOS-RF': oos_cm_results["rf"],
                              'OOS-NPV': oos_cm_results["npv"],
                              'OOS-MCC': oos_cm_results["mcc"],
                              'OOS-Recall': np.mean(recall_scores_oos),
                              'OOS-F1': np.mean(f1_scores_oos),
                              'OOS-EV': oos_ev*100
                              }
                            )
        
        return model_results, model
    
class TimeSeriesSplitImproved(TimeSeriesSplit):
    """Time Series cross-validator
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide `.
    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.
    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(n_splits=3)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [1] TEST: [2]
    TRAIN: [2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True,
    ...     train_splits=2):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [1 2] TEST: [3]
 
    Notes
    -----
    When ``fixed_length`` is ``False``, the training set has size
    ``i * train_splits * n_samples // (n_splits + 1) + n_samples %
    (n_splits + 1)`` in the ``i``th split, with a test set of size
    ``n_samples//(n_splits + 1) * test_splits``, where ``n_samples``
    is the number of samples. If fixed_length is True, replace ``i``
    in the above formulation with 1, and ignore ``n_samples %
    (n_splits + 1)`` except for the first training set. The number
    of test sets is ``n_splits + 2 - train_splits - test_splits``.
    """
 
    def split(self, X, y=None, groups=None, fixed_length=True, train_splits=4, test_splits=1):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        fixed_length : bool, hether training sets should always have
            common length
        train_splits : positive int, for the minimum number of
            splits to include in training sets
        test_splits : positive int, for the number of splits to
            include in the test set
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(("Cannot have number of folds ={0} greater than the number of samples: {1}.").format(n_folds,n_samples))
        if (n_folds - train_splits - test_splits) < 0 and test_splits > 0:
            raise ValueError("Both train_splits and test_splits must be positive integers.")
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds, n_samples - (test_size - split_size), split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)), test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],
                    indices[test_start:test_start + test_size])

if __name__ == "__main__":
    from Code.lib.plot_utils import PlotUtility
    from Code.lib.retrieve_data import DataRetrieve, ComputeTarget
    from Code.lib.time_utils import TimeUtility
    from Code.lib.feature_generator import FeatureGenerator
    from Code.utilities.stat_tests import stationarity_tests
#    from Code.lib.config import current_feature, feature_dict
    
    plotIt = PlotUtility()
    ct = ComputeTarget()
    dSet = DataRetrieve()
    featureGen = FeatureGenerator()
    timeUtil = TimeUtility()
    modelUtil = ModelUtility()
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

    modelname = "RF"
    model_results = []
    
    ######################
    # ML section
    ######################
    #  Make 'iterations' index vectors for the train-test split
    iterations = 200
         
    dX, dy = modelUtil.prepare_for_classification(mmData)        
    
    sss = StratifiedShuffleSplit(n_splits=iterations,
                                 test_size=0.33,
                                 random_state=None
                                 )
    
    tscv = TimeSeriesSplit(n_splits=6, max_train_size=24)
    tscvi = TimeSeriesSplitImproved(n_splits=6)
            
    model = RandomForestClassifier(n_jobs=-1,
                                   random_state=55,
                                   min_samples_split=10,
                                   n_estimators=500,
                                   max_features = 'auto',
                                   min_samples_leaf = 3,
                                   oob_score = 'TRUE'
                                   )
        
    ### add issue, other data OR use dictionary to pass data!!!!!!
    info_dict = {'issue':issue, 'modelStartDate':modelStartDate, 'modelEndDate':modelEndDate, 'modelname':modelname, 'nrows':nrows}
    model_results, fit_model = modelUtil.model_and_test(dX, dy, model, model_results, tscv, info_dict, evData)
#    print(model_results)
#    print(fit_model)
