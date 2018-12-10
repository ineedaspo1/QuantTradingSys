# -*- coding: utf-8 -*-
from numpy.random import randn
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from pylab import plot, show
import sys
from Code.lib.retrieve_data import DataRetrieve, ComputeTarget
import statsmodels.tsa.stattools as ts
"""
Created on Wed May  9 14:07:05 2018

@author: kruegkj

statistical tests
"""

def adf_test(df, signal, issue):
    print('\n====== ADF Test for Stationarity for ',signal, "=======")
    print('Issue: ', issue)
    print('Start Date: ', df.index.min().strftime('%Y-%m-%d'))
    print('End Date: ', df.index.max().strftime('%Y-%m-%d'))
    # Add print of issue
    # Add interpretation of meaning
    # Add print of time frame
    result = ts.adfuller(df[signal], 1)
    # p-value > 0.05: Accept the null hypothesis (H0), the data has a unit root and is non-stationary.
    # p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
    print('\nADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] < 0.05:
        print('** The series is likely stationary **')
    else:
        print('** The series is likely non-stationary **')
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


def hurst_setup(df, issue):
    # Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
    gbm = log(cumsum(randn(100000))+1000)
    mr = log(randn(100000)+1000)
    tr = log(cumsum(randn(100000)+1)+1000)

    # Output the Hurst Exponent for each of the above series
    #   H <0.5 = mean reverting
    #   H == 0.5 = random walk
    #   H >0.5 = momentum
    print('\n====== Hurst Exponent Test ======')
    print('Hurst(GBM):   %.3f' % hurst(gbm))
    print('Hurst(MR):    %.3f' % hurst(mr))
    print('Hurst(TR):    %.3f' % hurst(tr))
    print('Hurst(%s):   %.3f' % (issue, hurst(df)))

def stationarity_tests(s_df, signal, issue):
    print("=============================================================")
    adf_test(s_df, signal, issue)
    hurst_setup(s_df[signal][:], issue)
    print("========================================")

def mean_and_variance(value_series):
    # mean and varianace of series
    split = int(len(value_series) / 2)
    X1, X2 = value_series[0:split], value_series[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print("\n")
    print('mean1=%.2f, mean2=%.2f' % (mean1, mean2))
    print('variance1=%.2f, variance2=%.2f' % (var1, var2))


if __name__ == "__main__":
    dSet = DataRetrieve()
    ct = ComputeTarget()
    issue = "xly"
    dataLoadStartDate = "2015-01-01"
    dataLoadEndDate = "2016-03-30"

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    dataSet = dSet.read_issue_data(issue)
    adfDataSet = dSet.set_date_range(dataSet,
                                     dataLoadStartDate,
                                     dataLoadEndDate
                                     )
    
    #set beLong level
    beLongThreshold = 0.0
    adfDataSet = ct.setTarget(dataSet, "Long", beLongThreshold)

    stationarity_tests(adfDataSet, 'Close', issue)
    
    mean_and_variance(adfDataSet['beLong'][:])

    # dataSet.Close.hist()
