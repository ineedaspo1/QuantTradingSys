# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:03:09 2018
@author: KRUEGKJ
transformers.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy.stats import norm
from config import *

class Transformers:
    """Various signal transformation functions"""
    def zScore(self, df, ind, lb):
        """Used by zScore_transform for zScore calculation
            Args:
                p: the series having its z-score computed.
                lb: the lookback period, an integer.
                    the length used for the average and standard deviation.
                    typical values 3 to 10.
            Return:
                Return is a numpy array with values as z-scores
        """
        col_name = str(ind) + '_zScore_' + str(lb)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        nrows = df.shape[0]
        st = np.zeros(nrows)
        ma = np.zeros(nrows)
        p = df[ind]
        # use the pandas sliding window functions.
        st = p.rolling(window=lb, center=False).std()
        ma = p.rolling(window=lb, center=False).mean()
        z = np.zeros(nrows)
        for i in range(lb, nrows):
            z[i] = (p[i] - ma[i]) / st[i]
        df[col_name] = z
        return df

    def add_lag(self, df, lag_var, lags):
        """Add lag to any time-based series

            Args:
                df: dataframe column
                lag_var: indicator to be lagged
                lags: number of lags to add
                feature_dict: Dictionary of added features
            Returns:
                Dataframe with lagged indicator added
                feature_dict: Append entry with colname
            Usage:
                dataSet, feature_dict = transf.add_lag(
                        dataSet,
                        lag_var,
                        lags,
                        feature_dict)
        """
        for i in range(0, lags):
            df[lag_var + "_lag" + str(i+1)] = df[lag_var].shift(i+1)
            feature_dict[lag_var + "_lag" + str(i+1)] = 'Keep'
        return df

    def centering(self, df, col, lb=14, type='median'):
        """Subtract a historical median from signal

            Args:
                dataSet: Time series dataset
                col: Column name to be centered
                feature_dict: Dictionary of added features
                lb(int): lookback period
                type: default is median, could add more types in future
            Returns:
                dataSet: Dataset with new feature generated.
                feature_dict: Append entry with colname
            Usage:
                dataSet, feature_dict = transf.centering(
                        dataSet,
                        'ChaikinAD',
                        feature_dict,
                        14)
        """
        col_name = str(col) + '_Centered'
        feature_dict[col_name] = 'Keep'
        current_feature['Latest'] = col_name
        df[col_name] = df[col]
        if type == 'median':
            rm = df[col].rolling(window=lb, center=False).median()
            df[col_name] = df[col] - rm
        return df

    def scaler(self, df, col, type):
        """Scale - Use when sign and magnitude is of
                    paramount importance.
                    Scale accoring to historical volatility
                    defined by interquartile range.
            Args:
                df: Signal to be centered
                col: Column name to be centered
                feature_dict: Dictionary of added features
                lb(int): lookback period
                type: default is median, col
            Returns:
                dataSet: Dataset with new feature generated.
                feature_dict: Append entry with colname
            To Update: Expand code to call other scalers from sci-kit
        """
        col_name = str(col) + '_Scaled'
        feature_dict[col_name] = 'Keep'
        current_feature['Latest'] = col_name
        
        df = self.clean_dataset(df)
        
        df[col_name] = df[col]
        # current only coded to use RobustScaler
        scaler = RobustScaler(quantile_range=(25, 75))
        df[[col_name]] = scaler.fit_transform(df[[col_name]])
        return df

    def normalizer(self, dataSet, colname, n,
                   mode = 'scale', linear = False):
        """
             It computes the normalized value on the stats of n values
             ( Modes: total or scale ) using the formulas from the book
             "Statistically sound machine learning..." (Aronson and Masters)
             but the decission to apply a non linear scaling is left to the
             user. It's scale is supposed to be -100 to 100.
             -100 to 100 df is an imput DataFrame. it returns also a
             DataFrame, but it could return a list.

            Args:
                dataSet: dataframe cotaining signal to be normalized
                colname: Column name to be normalized
                n: number of data points to get the mean and the
                   quartiles for the normalization
                feature_dict: Dictionary of added features
                mode: scale: scale, without centering.
                        total: center and scale.
                linear: non-linear or linear scaling
            Returns:
                dataSet: Dataset with new feature generated.
                feature_dict: Append entry with colname
        """
        temp =[]
        new_colname = str(colname) + '_Normalized'
        feature_dict[new_colname] = 'Keep'
        current_feature['Latest'] = new_colname
        
        dataSet = self.clean_dataset(dataSet)
        
        df = dataSet[colname]
        for i in range(len(df))[::-1]:
            if i  >= n:
                # there will be a traveling norm until we reach the initial n
                # values. Those values will be normalized using the last
                # computed values of F50,F75 and F25
                F50 = df[i-n:i].quantile(0.5)
                F75 =  df[i-n:i].quantile(0.75)
                F25 =  df[i-n:i].quantile(0.25)
            if linear == True and mode == 'total':
                 v = 50 * ((df.iloc[i] - F50) / (F75 - F25)) - 50
            elif linear == True and mode == 'scale':
                 v =  25 * df.iloc[i] / (F75 - F25) -50
            elif linear == False and mode == 'scale':
                 v = 100 * norm.cdf(0.5 * df.iloc[i] / (F75 - F25)) - 50
            else:
                # even if strange values are given, it will perform full
                # normalization with compression as default
                v = norm.cdf(50*(df.iloc[i]-F50)/(F75-F25))-50
            temp.append(v)
        dataSet[new_colname] = temp[::-1]
        return  dataSet
    
    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df

if __name__ == "__main__":
    current_feature = {}
    from plot_utils import *
    from retrieve_data import *
    from config import *

    from oscillator_studies import *
    from ta_volume_studies import *
    from ta_momentum_studies import *
    from ta_volatility_studies import *

    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    
    taLibVolSt = TALibVolumeStudies()
    plotIt = PlotUtility()

    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)

    oscSt = OscialltorStudies()
    dataSet = oscSt.detrend_PO(dataSet, 'Close', 50)

    taLibMomSt = TALibMomentumStudies()
    dataSet = taLibMomSt.RSI(dataSet, 20)
    dataSet = taLibMomSt.rate_OfChg(dataSet, 50)
    
    vStud = TALibVolatilityStudies()
    dataSet = vStud.ATR(dataSet, 30)
    
    zScore_lb = 20
    transf = Transformers()
    transfList = list(feature_dict.keys())
    for i in transfList:
        print(i)
        dataSet= transf.zScore(dataSet, i, zScore_lb)

    # Plot price and indicators
    startDate = "2015-02-01"
    endDate = "2015-06-30"

    plotDataSet = dataSet[startDate:endDate]
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDataSet)

    ######################
    #### testing Lag
    feature_dict  = {}
    lag_var = 'Close'
    lags = 5
    dataSet = transf.add_lag(dataSet, lag_var, lags)
    # Plot price and lags
    startDate = "2015-02-01"
    endDate = "2015-04-30"
    lagDataSet = dataSet[startDate:endDate]

     # Set up plot dictionary
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, lagDataSet)

    ####################
    # Testing normalized, centered, scaled
    feature_dict  = {}

    dataSet = taLibMomSt.RSI(dataSet, 20)
    col_name = current_feature['Latest']
    print(col_name)
    
#    def clean_dataset(df):
#        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#        df.dropna(inplace=True)
#        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#        return df
    
    dataSet = transf.clean_dataset(dataSet)

    dataSet = transf.scaler(dataSet, col_name, 'robust')

    dataSet = transf.centering(dataSet, col_name, 14)

    dataSet = transf.normalizer(dataSet, col_name, 200, mode='scale',
                                linear=False
                                )

    startDate = "2015-10-01"
    endDate = "2016-06-30"
    normDataSet = dataSet[startDate:endDate]

     # Set up plot dictionary
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, normDataSet)
