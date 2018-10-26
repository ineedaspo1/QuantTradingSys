# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:32:53 2018

@author: KRUEGKJ

logreg_model_investigation.py
"""

import sys
sys.path.append('../lib')
sys.path.append('../utilities')

from plot_utils import *
from retrieve_data import *
from transformers import *
from feature_generator import *
from config import current_feature, feature_dict

# Import the Time Series library
#import statsmodels.tsa.stattools as ts
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import os.path

dSet = DataRetrieve()
transf = Transformers()
plotIt = PlotUtility()
featureGen = FeatureGenerator()  

issue = "XLV"

pivotDate = datetime.date(2012, 1, 1)
inSampleOutOfSampleRatio = 2
outOfSampleMonths = 6
inSampleMonths = inSampleOutOfSampleRatio * outOfSampleMonths
print("inSampleMonths: " + str(inSampleMonths))
segments = 1
months_to_load = outOfSampleMonths + segments * inSampleMonths
print("Months to load: " + str(months_to_load))
   
inSampleStartDate = pivotDate - relativedelta(months=months_to_load)
dataLoadStartDate = inSampleStartDate - relativedelta(months=1)
ooSampleStartDate = pivotDate - relativedelta(months=outOfSampleMonths)
print("Load Date: ", dataLoadStartDate)
print("In Sample Start  Date: ", inSampleStartDate)
print("Out Of Sample Start  Date: ", ooSampleStartDate)
print("Pivot Date: ", pivotDate)

dataSet = dSet.read_issue_data(issue)   
dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,pivotDate)

#dataSet = read_issue_data(issue, dataLoadStartDate, pivotDate)
print(issue)
nrows = dataSet.shape[0]
print ("nrows: ", nrows)

input_dict = {} # initialize 
#input_dict = {'f1': 
#              {'fname' : 'RSI', 
#               'params' : [10]
#               },
#              'f2': 
#              {'fname' : 'UltimateOscillator', 
#               'params' : [10 , 20, 30]
#               },
#              'f3': 
#              {'fname' : 'UltimateOscillator',
#               'params' : [],
#               'transform' : ['Normalized', 100]
#               },
#              'f4': 
#              {'fname' : 'RSI', 
#               'params' : [10],
#               'transform' : ['Zscore', 3]
#               },
#              'f5': 
#              {'fname' : 'RSI', 
#               'params' : [3],
#               'transform' : ['Scaler', 'robust']
#               },
#              'f6': 
#              {'fname' : 'RSI', 
#               'params' : [10],
#               'transform' : ['Center', 3]
#               },
#              'f7': 
#              {'fname' : 'Lag', 
#               'params' : ['Close', 3]
#               },
#              'f8': 
#              {'fname' : 'PPO', 
#               'params' : [12, 26]
#               },
#              'f9': 
#              {'fname' : 'CMO', 
#               'params' : [10]
#               },
#              'f10': 
#              {'fname' : 'CCI', 
#               'params' : [10]
#               },
#              'f11': 
#              {'fname' : 'ROC', 
#               'params' : [10]
#               },
#              'f12': 
#              {'fname' : 'ATR', 
#               'params' : [10]
#               },
#              'f13': 
#              {'fname' : 'NATR', 
#               'params' : [10]
#               },
#              'f14': 
#              {'fname' : 'ATRRatio', 
#               'params' : [10, 30]
#               },
#              'f15': 
#              {'fname' : 'DeltaATRRatio', 
#               'params' : [10, 50]
#               },
#              'f16': 
#              {'fname' : 'BBWidth', 
#               'params' : [10]
#               },
#              'f17': 
#              {'fname' : 'HigherClose', 
#               'params' : [4]
#               },
#              'f18': 
#              {'fname' : 'LowerClose', 
#               'params' : [4]
#               },
#              'f19': 
#              {'fname' : 'ChaikinAD', 
#               'params' : []
#               },
#              'f20': 
#              {'fname' : 'ChaikinADOSC', 
#               'params' : [4, 10],
#               'transform' : ['Normalized', 100]
#               },
#              'f21': 
#              {'fname' : 'OBV', 
#               'params' : [],
#               'transform' : ['Zscore', 3]
#               },
#              'f22': 
#              {'fname' : 'MFI', 
#               'params' : [14],
#               'transform' : ['Zscore', 3]
#               },
#              'f23': 
#              {'fname' : 'ease_OfMvmnt', 
#               'params' : [14],
#               'transform' : ['Zscore', 3]
#               },
#              'f24': 
#              {'fname' : 'exp_MA', 
#               'params' : [4]
#               },
#              'f25': 
#              {'fname' : 'simple_MA', 
#               'params' : [4]
#               },
#              'f26': 
#              {'fname' : 'weighted_MA', 
#               'params' : [4]
#               },
#              'f27': 
#              {'fname' : 'triple_EMA', 
#               'params' : [4]
#               },
#              'f28': 
#              {'fname' : 'triangMA', 
#               'params' : [4]
#               },
#              'f29': 
#              {'fname' : 'dblEMA', 
#               'params' : [4]
#               },
#              'f30': 
#              {'fname' : 'kaufman_AMA', 
#               'params' : [4]
#               },
#              'f31': 
#              {'fname' : 'delta_MESA_AMA', 
#               'params' : [0.9, 0.1],
#               'transform' : ['Normalized', 20]
#               },
#              'f32': 
#              {'fname' : 'inst_Trendline', 
#               'params' : []
#               },
#              'f33': 
#              {'fname' : 'mid_point', 
#               'params' : [4]
#               },
#              'f34': 
#              {'fname' : 'mid_price', 
#               'params' : [4]
#               },
#              'f35': 
#              {'fname' : 'pSAR', 
#               'params' : [4]
#               }
#             }
#input_dict = {'f1': 
#              {'fname' : 'RSI', 
#               'params' : [10],
#               'transform' : ['Scaler', 'robust']
#               },
#              'f2': 
#              {'fname' : 'delta_MESA_AMA', 
#               'params' : [0.9, 0.1],
#               'transform' : ['Scaler', 'robust']
#               }
#             }

#From Bandy book
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
              
                
df = featureGen.generate_features(dataSet, input_dict) 

predictor_vars = 'Fix this .... to inlcude feature names of model'
# Put indicators and transforms here

# set lag on Close (Pri)
#lag_var = 'ATR_5_zScore_10'
#lags = 3
#dataSet = transf.add_lag(dataSet, lag_var, lags)

# set lag on Close (Pri)
lag_var = 'RSI_2_Scaled'
lags = 4
dataSet = transf.add_lag(dataSet, lag_var, lags)

# set % return variables and lags
#dataSet["percReturn"] = dataSet["Close"].pct_change()*100
#lag_var = 'percReturn'
#lags = 3    
#dataSet = transf.add_lag(dataSet, lag_var, lags)

#set beLong level
beLongThreshold = 0
ct = ComputeTarget()
dataSet = ct.setTarget(dataSet, "Long", beLongThreshold)

col_name = "Volume"
current_feature['Latest'] = 'Volume'
dataSet = transf.normalizer(dataSet, col_name, 50, mode='scale', linear=False)

plot_dict = {}
plot_dict['Issue'] = issue
plot_dict['Plot_Vars'] = list(feature_dict.keys())
plot_dict['Volume'] = 'Yes'
plotIt.price_Ind_Vol_Plot(plot_dict, dataSet)

modelStartDate = inSampleStartDate
modelEndDate = modelStartDate + relativedelta(months=inSampleMonths)

mmData = dataSet[modelStartDate:modelEndDate].copy()

nrows = mmData.shape[0]
print ("beLong counts: ")
be_long_count = mmData['beLong'].value_counts()
print (be_long_count)
print ("out of ", nrows)

plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
plotIt.plot_v2x(mmData, plotTitle)
plotIt.histogram(mmData['beLong'], x_label="beLong signal", y_label="Frequency", 
  title = "beLong distribution for " + issue)  
plt.show(block=False)

# Chnage -1 to 0 for log reg
#mmData.beLong.replace([1, -1], [1, 0], inplace=True)
datay = mmData['beLong']
nrows = datay.shape[0]
print ("nrows beLong: ", nrows)

col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
to_drop = ['Open','High','Low', 'gainAhead', 'Symbol', 'Close', 'Date', 'beLong']
for x in to_drop:
    col_vals.append(x)
mmData = dSet.drop_columns(mmData, col_vals)

# from correlation study
#col_vals = ['ATR_5_zScore_10_lag1', 'ATR_5_zScore_10_lag3']
#mmData = dSet.drop_columns(mmData, col_vals)

#print(list(mmData.columns.values))

dataX = mmData

# get names of features
names = dataX.columns.values.tolist()

# correlation plot
#def correlation_matrix(df,size=10):
#    from matplotlib import pyplot as plt
#    from matplotlib import cm as cm
#    fig = plt.figure(figsize=(size, size))
#    ax1 = fig.add_subplot(111)
#    cmap = cm.get_cmap('jet', 30)
#    corr = df.corr()
#    cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
#    ax1.grid(True)
#    plt.title('Feature Correlation')
#    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
#    plt.yticks(range(len(corr.columns)), corr.columns);
#    # Add colorbar, make sure to specify tick locations to match desired ticklabels
#    fig.colorbar(cax, ticks=[-1, -.5, 0, .5 ,1])
#    plt.show()
#    
#    c1 = corr.abs().unstack()
#    print(c1.sort_values(ascending = False))
#correlation_matrix(dataX)


#  Copy from pandas dataframe to numpy arrays
dy = np.zeros_like(datay)
dX = np.zeros_like(dataX)

dy = datay.values
dX = dataX.values

#########################################
#import statsmodels.api as sm
#logit_model=sm.Logit(datay,dataX)
#result=logit_model.fit()
#print(result.summary())
#####################################

######################
# ML section
model_results = []
iterations = 300

from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors=8) 
modelname = 'KNN'

#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_jobs=-1, random_state=0, min_samples_split=20, n_estimators=500, max_features = 'auto', min_samples_leaf = 20, oob_score = 'TRUE')
#modelname = 'RF'

#C = 2
#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(C= C)
#modelname = 'LR'

#  Make 'iterations' index vectors for the train-test split
sss = StratifiedShuffleSplit(n_splits=iterations,test_size=0.33, random_state=None)

accuracy_scores_is = []
accuracy_scores_oos = []
precision_scores_is = []
precision_scores_oos = []
recall_scores_is = []
recall_scores_oos = []
f1_scores_is = []
f1_scores_oos = []

#  Initialize the confusion matrix
cm_sum_is = np.zeros((2,2))
cm_sum_oos = np.zeros((2,2))
    
#  For each entry in the set of splits, fit and predict
for train_index,test_index in sss.split(dX,dy):
    X_train, X_test = dX[train_index], dX[test_index]
    y_train, y_test = dy[train_index], dy[test_index] 
    
#  fit the model to the in-sample data
    model.fit(X_train, y_train)

#  test the in-sample fit       
    y_pred_is = model.predict(X_train)
    #for i in range(0, 5):
    #    print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_train)[i], y_pred_is[i]))
    cm_is = confusion_matrix(y_train, y_pred_is)
    cm_sum_is = cm_sum_is + cm_is
    accuracy_scores_is.append(accuracy_score(y_train, y_pred_is))
    precision_scores_is.append(precision_score(y_train, y_pred_is))
    recall_scores_is.append(recall_score(y_train, y_pred_is))
    f1_scores_is.append(f1_score(y_train, y_pred_is))
    
#  test the out-of-sample data
    y_pred_oos = model.predict(X_test)
    #for i in range(0, 5):
    #    print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], y_pred_oos[i]))
    cm_oos = confusion_matrix(y_test, y_pred_oos)
    cm_sum_oos = cm_sum_oos + cm_oos
    accuracy_scores_oos.append(accuracy_score(y_test, y_pred_oos))
    precision_scores_oos.append(precision_score(y_test, y_pred_oos))
    recall_scores_oos.append(recall_score(y_test, y_pred_oos))
    f1_scores_oos.append(f1_score(y_test, y_pred_oos))

#print (sorted(zip(map(lambda x: round(x, 2), model.feature_importances_), names), reverse=True))

tpIS = cm_sum_is[1,1]
fnIS = cm_sum_is[1,0]
fpIS = cm_sum_is[0,1]
tnIS = cm_sum_is[0,0]
precisionIS = tpIS/(tpIS+fpIS)
recallIS = tpIS/(tpIS+fnIS)
accuracyIS = (tpIS+tnIS)/(tpIS+fnIS+fpIS+tnIS)
f1IS = (2.0 * precisionIS * recallIS) / (precisionIS+recallIS) 

tpOOS = cm_sum_oos[1,1]
fnOOS = cm_sum_oos[1,0]
fpOOS = cm_sum_oos[0,1]
tnOOS = cm_sum_oos[0,0]
precisionOOS = tpOOS/(tpOOS+fpOOS)
recallOOS = tpOOS/(tpOOS+fnOOS)
accuracyOOS = (tpOOS+tnOOS)/(tpOOS+fnOOS+fpOOS+tnOOS)
f1OOS = (2.0 * precisionOOS * recallOOS) / (precisionOOS+recallOOS) 

print ("\n\nSymbol is ", issue)
print ("Learning algorithm is ", model)
print ("Confusion matrix for %i randomized tests" % iterations)
print ("for dates ", modelStartDate, " through ", modelEndDate) 
print ("----------------------")
print ("==In Sample==")
print ("     Predicted")
print ("      pos neg")
print ("pos:  %i  %i  Recall:%.2f" % (tpIS, fnIS, recallIS))
print ("neg:  %i  %i" % (fpIS, tnIS))
print ("Prec: %.2f          Accuracy: %.2f " % (precisionIS, accuracyIS))
print ("f1:   %.2f" % f1IS)
print ("\n----------------------")
print ("\n==Out Of Sample==")
print ("     Predicted")
print ("      pos neg")
print ("pos:  %i  %i  Recall:%.2f" % (tpOOS, fnOOS, recallOOS))
print ("neg:  %i  %i" % (fpOOS, tnOOS))
print ("Prec: %.2f          Accuracy: %.2f " % (precisionOOS, accuracyOOS))
print ("f1:   %.2f" % f1OOS)
print ("\n----------------------")

print ("\n==In Sample==")
print('Accuracy: %.2f' % np.mean(accuracy_scores_is))
print('Precision: %.2f' % np.mean(precision_scores_is))
print('Recall: %.2f' % np.mean(recall_scores_is))
print('F1: %.2f' % np.mean(f1_scores_is))
print ("\n==Out Of Sample==")
print('Accuracy: %.2f' % np.mean(accuracy_scores_oos))
print('Precision: %.2f' % np.mean(precision_scores_oos))
print('Recall: %.2f' % np.mean(recall_scores_oos))
print('F1: %.2f' % np.mean(f1_scores_oos))
print ("\nend of run")

model_results.append({'Issue': issue, 'StartDate': modelStartDate.strftime("%Y-%m-%d"), 'EndDate': modelEndDate.strftime("%Y-%m-%d"), 'Model': modelname, 'Rows': nrows, 'beLongCount': str(np.count_nonzero(dy==1)), 'Predictors': predictor_vars, 'IS-Accuracy': np.mean(accuracy_scores_is), 'IS-Precision': np.mean(precision_scores_is), 'IS-Recall': np.mean(recall_scores_is), 'IS-F1': np.mean(f1_scores_is), 'OOS-Accuracy':  np.mean(accuracy_scores_oos), 'OOS-Precision': np.mean(precision_scores_oos), 'OOS-Recall': np.mean(recall_scores_oos), 'OOS-F1': np.mean(f1_scores_oos)})
        
df = pd.DataFrame(model_results)
df = df[['Issue','StartDate','EndDate','Model','Rows','beLongCount','Predictors','IS-Accuracy','IS-Precision','IS-Recall','IS-F1','OOS-Accuracy','OOS-Precision','OOS-Recall','OOS-F1']]
print(df)

dirext = issue + '_Model_' + modelname + '_start_' + str(dataLoadStartDate) + '_end_' + str(pivotDate) + '_TS_' + datetime.datetime.now().strftime("%Y-%m-%d")
print(dirext)
filename = "IS_model_iteration_" + dirext + ".csv"
current_directory = os.getcwd()
df.to_csv(current_directory+"\\"+filename, encoding='utf-8', index=False)

"""
Adding validation code
"""
print("\n\n===================================================")
print("Walk Forward Validation")
print("In sample start date: ", modelStartDate)

# Select the date range
modelStartDate = ooSampleStartDate
modelEndDate = modelStartDate + relativedelta(months=outOfSampleMonths)
print("Out of sample start date: ", modelStartDate)
print("===================================================\n")
valData = dataSet[modelStartDate:modelEndDate].copy()
tradesData = valData.copy()
    
nrows = valData.shape[0]
print ("beLong counts: ")
be_long_count = valData['beLong'].value_counts()
print (be_long_count)
print ("out of ", nrows)

plotTitle = issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
plotIt.plot_v2x(valData, plotTitle)
plotIt.histogram(valData['beLong'], x_label="beLong signal", y_label="Frequency", 
  title = "beLong distribution for " + issue)        
plt.show(block=False)

valModelData = valData.copy()
valModelData = dSet.drop_columns(valModelData, col_vals)

valRows = valModelData.shape[0]
#print("There are %i data points" % valRows)

# test the validation data
y_validate = []
y_validate = model.predict(valModelData)

# Create best estimate of trades
bestEstimate = np.zeros(valRows)

# You may need to adjust for the first and / or final entry 

for i in range(1,valRows -1):
    print('{0} {1:8.2%} {2:>16}'.format(valData.Date.iloc[i].strftime('%Y-%m-%d'), valData.gainAhead.iloc[i], y_validate[i]))
    if y_validate[i] > 0.0: 
        bestEstimate[i] = valData.gainAhead.iloc[i]
    else:
        bestEstimate[i] = 0.0 

# Create and plot equity curve
equity = np.zeros(valRows)
equity[0] = 1.0
for i in range(1,valRows):
    equity[i] = (1+bestEstimate[i])*equity[i-1]
print('\n')    
print('{0:<10s} {1:10} {2:>20s} {3:>10} {4:>10}'.format('Date', 'Gain Ahead', 'Predict (beLong)', 'Best estimate', 'Equity'))
for i in range(1,valRows-1):
    print('{0} {1:12.3f} {2:>16} {3:10.2%} {4:14.3f}'.format(valData.Date.iloc[i].strftime('%Y-%m-%d'), valData.gainAhead.iloc[i], y_validate[i],bestEstimate[i],equity[i]))
    
print("\nTerminal Weatlh: ", equity[valRows-1])
plt.plot(equity)

print("\n ================ End of Run =================")

valData['valBeLong'] = pd.Series(y_validate, index=valData.index)

#==========================
import matplotlib as mpl
#from matplotlib import style
plt.style.use('seaborn-ticks')
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(hspace=0.05)
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((6,1), (4,0), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1)

ax2.plot(valData['valBeLong'], color='green', alpha =0.6)
ax1.plot(valData['Close'])
ax3.plot(valData['beLong'], color='purple', alpha =0.6)

ax1.label_outer()
ax2.label_outer()
ax2.tick_params(axis='x',which='major',bottom='on')
ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax2.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax3.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
ax2.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
ax3.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)
ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax1.label_outer()
ax1.legend(loc='upper left', frameon=True, fontsize=8)
ax2.label_outer()
ax2.legend(loc='upper left', frameon=True, fontsize=8)
ax3.label_outer()
ax3.legend(loc='upper left', frameon=True, fontsize=8)

#==========================
tradesData['valBeLong'] = pd.Series(y_validate, index=tradesData.index)
tradesData['gain'] = tradesData['Close'] - tradesData['Open']

#  Count the number of rows in the file
nrows = tradesData.shape[0]
print ('There are %0.f rows of data' % nrows)

#  Compute cumulative equity for all days
equityAllSignals = np.zeros(nrows)
equityAllSignals[0] = 1
for i in range(1,nrows):
    equityAllSignals[i] = (1+tradesData.gainAhead[i])*equityAllSignals[i-1]

print ('TWR for all signals is %0.3f' % equityAllSignals[nrows-1])
# add to valData
valData['equityAllSignals'] = pd.Series(equityAllSignals, index=valData.index)
    
#  Compute cumulative equity for days with beLong signals    
equityBeLongSignals = np.zeros(nrows)
equityBeLongSignals[0] = 1
for i in range(1,nrows):
    if (tradesData.beLong[i] > 0):
        equityBeLongSignals[i] = (1+tradesData.gainAhead[i])*equityBeLongSignals[i-1]
    else:
        equityBeLongSignals[i] = equityBeLongSignals[i-1]
valData['equityBeLongSignals'] = pd.Series(equityBeLongSignals, index=valData.index)

#  Compute cumulative equity for days with Validation beLong signals    
equityValBeLongSignals = np.zeros(nrows)
equityValBeLongSignals[0] = 1
for i in range(1,nrows):
    if (tradesData.valBeLong[i] > 0):
        equityValBeLongSignals[i] = (1+tradesData.gainAhead[i])*equityValBeLongSignals[i-1]
    else:
        equityValBeLongSignals[i] = equityValBeLongSignals[i-1]
               
print ('TWR for all days with beLong signals is %0.3f' % equityBeLongSignals[nrows-1])
valData['equityValBeLongSignals'] = pd.Series(equityValBeLongSignals, index=valData.index)

#  Plot the two equity streams
fig = plt.figure(figsize=(12,8))
fig.suptitle(issue + ' Portfolio value in Validation')
ax1 = fig.add_subplot(111)
ax1.plot(valData.equityBeLongSignals, color='green',label='BeLong (Max potential)')
ax1.plot(valData.equityAllSignals, color='blue',label='Cumulative Equity, All days')
ax1.plot(valData.equityValBeLongSignals, color='purple',label='Cumulative, Equity, Predict')

ax1.legend(loc='upper left', frameon=True, fontsize=8)
ax1.label_outer()
ax1.tick_params(axis='x',which='major',bottom='on')
ax1.minorticks_on()
ax1.grid(True, which='major', color='k', linestyle='-', alpha=0.6)
ax1.grid(True, which='minor', color='r', linestyle='-', alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(valData.Close, color='black',alpha=0.6,label='CLOSE',linestyle='--')
ax2.legend(loc='center left', frameon=True, fontsize=8)
ax2.label_outer()
plt.show()
#
#
#
#
#===================================        
# Getting sample equity curve

#  Evaluation of signals
print ("\n\n===================================")
print ("Starting single run")
print ("===================================")

#  Status variables
ndays = tradesData.shape[0]

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
predSignal = np.zeros(ndays)
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
predSignal[0] = 0

#  Loop over all the days loaded
for i in range (1,ndays):
    #  Extract the date
    dt = tradesData.index[i]
    #  Check the date
    datesPass = dt.date()>=modelStartDate and dt.date()<=modelEndDate
    if datesPass:
        iTradeDay = iTradeDay + 1
        
        if sharesHeld[iTradeDay-1] > 0:
            #  In a long position
            if tradesData.valBeLong[i]<0:
                #  target is -1 -- beFlat 
                #  Exit -- close the trade
                exitPrice = tradesData.Close[i]
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
                predSignal[i] = "-1"
                
                pass
            else:
                #  target is +1 -- beLong
                #  Continue long
                sharesHeld[iTradeDay] = sharesHeld[iTradeDay-1]
                cash[iTradeDay] = cash[iTradeDay-1]
                MTMPrice = tradesData.Close[i]
                openTradeEquity = sharesHeld[iTradeDay] * MTMPrice
                accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                predSignal[i] = "0"
                pass
        else:
            #  Currently flat
            if tradesData.valBeLong[i]>0:
                #  target is +1 -- beLong
                #  Enter a new position
                entryPrice = tradesData.Close[i]
                sharesHeld[iTradeDay] = int(fixedTradeDollars/(entryPrice+commission))
                shareCost = sharesHeld[iTradeDay]*(entryPrice+commission)
                cash[iTradeDay] = cash[iTradeDay-1] - shareCost
                openTradeEquity = sharesHeld[iTradeDay]*entryPrice
                accountBalance[iTradeDay] = cash[iTradeDay] + openTradeEquity
                predSignal[i] = "1"
                pass
            else:
                #  target is -1 -- beFlat
                #  Continue flat
                cash[iTradeDay] = cash[iTradeDay-1]
                accountBalance[iTradeDay] = cash[iTradeDay]
                predSignal[i] = "0"
                pass
            
tradesData['accountBalance'] = pd.Series(accountBalance, index=tradesData.index)
tradesData['sharesHeld'] = pd.Series(sharesHeld, index=tradesData.index)
tradesData['cash'] = pd.Series(cash, index=tradesData.index)
tradesData['predSignal'] = pd.Series(predSignal, index=tradesData.index)


fig, ax = plt.subplots(figsize=(14,8))
buys = tradesData.ix[(tradesData['predSignal'] == 1)]
sells = tradesData.ix[(tradesData['predSignal'] == -1)]

ax.plot(tradesData.index, tradesData['Close'])
ax.plot(buys.index, tradesData.ix[buys.index]['Close'], '^', markersize=6, color='g', label='Buy')
ax.plot(sells.index, tradesData.ix[sells.index]['Close'], 'v', markersize=6, color='r', label='Sell')

plotTitle = "Predictive Buy/Sell signals for " + issue + ", " + str(modelStartDate) + " to " + str(modelEndDate)
fig.suptitle(plotTitle)
fig.autofmt_xdate()
ax.label_outer()
ax.legend(loc='upper left', frameon=True, fontsize=8)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.grid(True, which='both')
ax.xaxis_date()
ax.autoscale_view()
ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
ax.minorticks_on()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f')) 
plt.show()

#  Format and print results        
print('Starting equity: %.2f' % initialEquity)
finalAccountBalance = accountBalance[iTradeDay]
print ('Final account balance: %.2f' %  finalAccountBalance)
print ("Number of trading days:", iTradeDay)      
numberTrades = iTradeNumber
print ("Number of trades:", numberTrades)

from pandas import Series

Sequity = Series(accountBalance[0:numberTrades-1])

Sequity.plot(title="Account balance")

tradeWins = sum(1 for x in tradeGain if float(x) >= 1.0)
tradeLosses = sum(1 for x in tradeGain if float(x) < 1.0 and float(x) > 0)
print("Wins: ", tradeWins)
print("Losses: ", tradeLosses)
print('W/L: %.2f' % (tradeWins/numberTrades))

tradeWinsValue = sum((x*fixedTradeDollars)-fixedTradeDollars for x in tradeGain if float(x) >= 1.0)
tradeLossesValue = sum((x*fixedTradeDollars)-fixedTradeDollars for x in tradeGain if float(x) < 1.0 and float(x) > 0)
print('Total value of Wins:  %.2f' % tradeWinsValue)
print('Average win: %.2f' % (tradeWinsValue / tradeWins))
print('Total value of Losses:  %.2f' % tradeLossesValue)
print('Average loss: %.2f' % (tradeLossesValue/tradeLosses))
#(Win % x Average Win Size) – (Loss % x Average Loss Size)
print('Expectancy:  %.2f' % ((tradeWins/numberTrades)*(tradeWinsValue/tradeWins)-(tradeLosses/numberTrades)*(tradeLossesValue/tradeLosses)))
print("Fixed trade size: ", fixedTradeDollars)

# Expectancy = (AW × PW + AL × PL) ⁄ |AL|
# AW = average winning trade (excluding maximum win)
AW =  tradeWinsValue / tradeWins # remove max win
# PW = <wins> ⁄ NST
PW = tradeWins / numberTrades
# AL = average losing trade (negative, excluding scratch losses)
AL = tradeLossesValue / tradeLosses
# PL = probability of losing: PL = <non-scratch losses> ⁄ NST 
PL = tradeLosses / numberTrades

print('AW: %.2f' % AW)
print('PW: %.2f' % PW)
print('AL: %.2f' % AL)
print('PL: %.2f' % PL)

Expectancy = (AW * PW + AL * PL)/ abs(AL)
print('Expectancy: %.2f' % Expectancy)

# Sharpe ratio...probably not correct math
#import math
#print(np.mean(tradeGainDollars))
#print(np.std(tradeGainDollars))
#print(math.sqrt(numberTradeDays)*(np.mean(tradeGainDollars)/np.std(tradeGainDollars)))


####  end  ####     
df_to_save = tradesData[['valBeLong','gainAhead']].copy()
df_to_save.reset_index(level=df_to_save.index.names, inplace=True)
df_to_save.columns=['Date','signal','gainAhead']
#print(df_to_save)

dirext = issue + '_test1'
filename = "oos_equity_eval_" + dirext + ".csv" 
df_to_save.to_csv(filename, encoding='utf-8', index=False)
