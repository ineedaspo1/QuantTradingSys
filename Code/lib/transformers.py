# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:03:09 2018
@author: KRUEGKJ
transformers.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from Code.lib.config import current_feature, feature_dict
from pprint import pprint

'''Normalization is a rescaling of the data from the original range so that all values are within the range of 0 and 1'''

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
        feature_dict[current_feature['Latest']] = 'Drop'
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

    def add_lag(self, df, lag=2):
        """Add lag to any time-based series

            Args:
                df: dataframe column
                lag_var: indicator to be lagged
                lag: distance to lag
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
        # hard coded use of 'Close' for now
        lag_var = 'Close'
        col_name = lag_var + "_lag" + str(lag)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = df[lag_var].shift(-lag)
        #print(df[col_name].tail(20))
        df[col_name] = df[col_name].fillna(df[col_name].rolling(lag*2,center=True,min_periods=1).mean())
#        df['rollmean3']  = df[col_name].rolling(1,center=True,min_periods=1).mean()
#        df['update'] = df['rollmean3']
#        df['update'].update( df[col_name] )
        #print("===============")
        #print(df.tail(20))
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
        feature_dict[current_feature['Latest']] = 'Drop'
        current_feature['Latest'] = col_name
        df[col_name] = df[col]
        
        if type == 'median':
            rm = df[col].rolling(window=lb, center=False).median()
            df[col_name] = df[col] - rm
        # scale values skipping nan's
        scaler = MinMaxScaler(feature_range=(0, 1))
        null_index = df[col_name].isnull()
        df.loc[~null_index, [col_name]] = scaler.fit_transform(df.loc[~null_index, [col_name]])
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
        feature_dict[current_feature['Latest']] = 'Drop'
        current_feature['Latest'] = col_name
            
        df = self.clean_dataset(df)
        
        df[col_name] = df[col]
        # current only coded to use RobustScaler
        scaler = RobustScaler(quantile_range=(25, 75))
        df[[col_name]] = scaler.fit_transform(df[[col_name]])
        # scale values skipping nan's
        scaler = MinMaxScaler(feature_range=(-1, 1))
        null_index = df[col_name].isnull()
        df.loc[~null_index, [col_name]] = scaler.fit_transform(df.loc[~null_index, [col_name]])
        return df

    def normalizer(self, dataSet, colname, n,
                   mode = 'total', linear = False):
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
        """
        temp =[]
        new_colname = str(colname) + '_Normalized_' + str(n)
        feature_dict[new_colname] = 'Keep'
        #print(feature_dict)
        feature_dict[current_feature['Latest']] = 'Drop'
        #print(feature_dict)
        current_feature['Latest'] = new_colname
        # clean NaN's
#        print("tail before clean: ")
#        pprint(dataSet.tail(10))
        dataSet = self.clean_dataset(dataSet)
#        print("tail after clean: ")
#        pprint(dataSet.tail(10))
        
        df = dataSet[colname]
        #print(df)
        for i in range(len(df))[::-1]:
            #print("i: " + str(i) + "    n: " + str(n))
            if i  >= n:
                # there will be a traveling norm until we reach the initial n
                # values. Those values will be normalized using the last
                # computed values of F50,F75 and F25
                F50 = df[i-n:i].quantile(0.5)
                F75 =  df[i-n:i].quantile(0.75)
                F25 =  df[i-n:i].quantile(0.25)
            if linear == True and mode == 'total':
                 v = 0.5 * ((df.iloc[i] - F50) / (F75 - F25)) - 0.5
            elif linear == True and mode == 'scale':
                 v =  0.25 * df.iloc[i] / (F75 - F25) - 0.5
            elif linear == False and mode == 'scale':
                 v = 0.5 * norm.cdf(0.5 * df.iloc[i] / (F75 - F25)) - 0.5
            else:
                # even if strange values are given, it will perform full
                # normalization with compression as default
                
                v = norm.cdf(1.0*(df.iloc[i]-F50)/(F75-F25))
            #print (v)
            temp.append(v)
        dataSet[new_colname] = temp[::-1]
        return  dataSet
    
    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df

if __name__ == "__main__":
    from Code.lib.plot_utils import PlotUtility
    from Code.lib.retrieve_data import DataRetrieve
    from Code.lib.transformers import Transformers
    from Code.lib.oscillator_studies import OscialltorStudies
    from Code.lib.ta_momentum_studies import TALibMomentumStudies
    from Code.lib.ta_volatility_studies import TALibVolatilityStudies
    from Code.lib.feature_generator import FeatureGenerator
    from Code.lib.config import current_feature, feature_dict
    
    vStud = TALibVolatilityStudies()
    taLibMomSt = TALibMomentumStudies()
    plotIt = PlotUtility()
    oscSt = OscialltorStudies()
    dSet = DataRetrieve()
    transf = Transformers()
    featureGen = FeatureGenerator()
    
    issue = "TLT"
    df = dSet.read_issue_data(issue)
    dataLoadStartDate = df.Date[0]
    pprint(dataLoadStartDate)
    lastRow = df.shape[0]
    dataLoadEndDate = df.Date[lastRow-1]
    pprint(dataLoadEndDate)
    df = dSet.set_date_range(df, dataLoadStartDate,dataLoadEndDate)
    pprint(df.tail(10))
    # Resolve any NA's for now
    df.fillna(method='ffill', inplace=True)
    
    input_dict = {} # initialize 
    input_dict = {'f1': 
                  {'fname' : 'OBV', 
                   'params' : [9],
                   'transform':  ['Scaler', 'Robust']
                   },
                  'f2': 
                  {'fname' : 'OBV', 
                   'params' : [],
                   'transform':  ['Normalized', 10]
                   }
                  }
    
    dataSet = featureGen.generate_features(df, input_dict)
    
    #dataSet = transf.normalizer(dataSet, 'Volume', 50)
            
    # Plot price and indicators
    startDate = "2019-01-20"
    endDate = "2019-02-12"

    plotDataSet = dataSet[startDate:endDate]
    
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDataSet)
    