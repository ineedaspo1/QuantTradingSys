# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:07:05 2018

@author: kruegkj

statistical tests
"""
from __future__ import print_function

# Import the Time Series library
import statsmodels.tsa.stattools as ts
from retrieve_issue_data import *
from pylab import plot, show
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import matplotlib.mlab as mlab
import pandas
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 11, 5

def adf_test(df, issue):
    print ('\n====== ADF Test for Stationarity ======')
    print ('Issue: ', issue)
    print ('Start Date: ', df.index.min().strftime('%Y-%m-%d'))
    print ('End Date: ', df.index.max().strftime('%Y-%m-%d'))
    # Add print of issue
    # Add interpretation of meaning
    # Add print of time frame
    result = ts.adfuller(df['Close'],autolag='AIC')
    # p-value > 0.05: Accept the null hypothesis (H0), the data has a unit root and is non-stationary.
    # p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
    print('\nADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] < 0.05:
        print ('** The series is likely stationary **')
    else:
        print ('** The series is likely non-stationary **')
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
def hurst(df):
    closes = df
    #plot(closes); show()
    # calculate Hurst
    lag1 = 2
    lags = range(lag1, 20)
    tau = [sqrt(std(subtract(closes[lag:], closes[:-lag]))) for lag in lags]
     
    #plot(log(lags), log(tau)); show()
    m = polyfit(log(lags), log(tau), 1)
    hurst = m[0]*2
    return hurst
    
def hurst_setup(df,issue):
    # Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
    gbm = log(cumsum(randn(100000))+1000)
    mr = log(randn(100000)+1000)
    tr = log(cumsum(randn(100000)+1)+1000)
    
    # Output the Hurst Exponent for each of the above series
    #   H <0.5 = mean reverting
    #   H == 0.5 = random walk 
    #   H >0.5 = momentum
    print ('\n====== Hurst Exponent Test ======')
    print ('Hurst(GBM):   %.3f' % hurst(gbm))
    print ('Hurst(MR):    %.3f' % hurst(mr))
    print ('Hurst(TR):    %.3f' % hurst(tr))
    print ('Hurst(%s):   %.3f' % (issue, hurst(df)))
    
if __name__ == "__main__":
    issue = "xly"
    dataLoadStartDate = "2017-11-01"
    dataLoadEndDate = "2018-03-30"  
    
    startDate = "2017-11-01"
    endDate = "2018-03-30"    
    dataSet = read_issue_data(issue, dataLoadStartDate, dataLoadEndDate)
    adfDataSet = dataSet.ix[startDate:endDate]
    
    adf_test(adfDataSet, issue)
    #hurst_setup(adfDataSet['Close'][:],issue)
    # histogram of prices
    plt.figure(2)   
    dataSet.Close.hist()

    plt.figure(3)   
    plt.plot(dataSet.Close)
    dClose = dataSet.Close.values
    # mean and varianace of series
    split = int(len(dClose) / 2)
    X1, X2 = dClose[0:split], dClose[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))
    
    lnprice=np.log(dataSet.Close)
    lnprice
    plt.figure(4)
    plt.plot(lnprice)
    plt.show()
    acf_1 =  acf(lnprice)[1:20]
    plt.plot(acf_1)
    test_df = pandas.DataFrame([acf_1]).T
    test_df.columns = ['Pandas Autocorrelation']
    test_df.index += 1
    test_df.plot(kind='bar')
    pacf_1 =  pacf(lnprice)[1:20]
    plt.plot(pacf_1)
    test_df = pandas.DataFrame([pacf_1]).T
    test_df.columns = ['Pandas Partial Autocorrelation']
    test_df.index += 1
    test_df.plot(kind='bar')
    result = ts.adfuller(lnprice, 1)
    result
    
    lnprice_diff=lnprice-lnprice.shift()
    diff=lnprice_diff.dropna()
    acf_1_diff =  acf(diff)[1:20]
    test_df = pandas.DataFrame([acf_1_diff]).T
    test_df.columns = ['First Difference Autocorrelation']
    test_df.index += 1
    test_df.plot(kind='bar')
    pacf_1_diff =  pacf(diff)[1:20]
    plt.plot(pacf_1_diff)
    plt.show()
    
    price_matrix=lnprice.as_matrix()
    model = ARIMA(price_matrix, order=(0,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    predictions=model_fit.predict(122, 127, typ='levels')
    predictions
    predictionsadjusted=np.exp(predictions)
    predictionsadjusted
    
    decomposition = sm.tsa.seasonal_decompose(dataSet.Close, model='additive')
    fig = decomposition.plot()
    plt.show()

    