# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018

@author: KRUEGKJ

ta_volume_studies.py
"""
import talib as ta
from Code.lib.config import current_feature, feature_dict

class TALibVolumeStudies:
    """Group of Volume studies utilized fromTALib """
    def ChaikinAD(self, df):
        """ChaikinAD The Chaikin Accumulation / Distribution (AD) 
        line is a measure of the money flowing into or out of a security. 
        It is similar to On Balance Volume (OBV).    
           Args:
                high, low, close: hlc of issue
                volume: volume of issue
                feature_dict: Dictionary of added features
           Return:
                chaikinAD
                feature_dict
        """
        col_name = 'ChaikinAD'
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        df[col_name] = ta.AD(df.High,
            df.Low,
            df.Close, 
            df.Volume
            )
        return df
    
    def ChaikinADOSC(self, df, fst_prd=5, slw_prd=30):
        """ChaikinADOSC The Chaikin Accumulation / Distribution (AD) Osc 
        moving average oscillator based on the Accumulation/Distribution
        indicator.    
           Args:
                high, low, close: hlc of issue
                volume: volume of issue
                fastperiod: Fast moving average
                slowperiod: Slow moving average
                feature_dict: Dictionary of added features
           Return:
                chaikinADOSC
                feature_dict
        """
        col_name = 'ChaikinADOSC_f' + str(fst_prd) + '_s' + str(slw_prd)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.ADOSC(df.High,
                                df.Low,
                                df.Close, 
                                df.Volume,
                                fst_prd,
                                slw_prd
                                )
        return df
    
    def OBV(self, df):
        """On Balance Volume (OBV) measures buying and selling pressure
        as a cumulative indicator that adds volume on up days and 
        subtracts volume on down days.    
           Args:
                close: close of issue
                volume: volume of issue
                feature_dict: Dictionary of added features
           Return:
                onBalVol
                feature_dict
        """
        col_name = 'OBV'
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.OBV(df.Close, 
                              df.Volume
                              )
        return df
    
    def MFI(self, df2, period):
        """Money Flow Index (MFI)
        Uses both price and volume to measure buying and selling pressure. It is
        positive when the typical price rises (buying pressure) and negative when
        the typical price declines (selling pressure). A ratio of positive and
        negative money flow is then plugged into an RSI formula to create an
        oscillator that moves between zero and one hundred.
        http://stockcharts.com/school/doku.php?
        id=chart_school:technical_indicators:money_flow_index_mfi
        Args:
            dataSet: Price series dataet
            n(int): n period.
        Returns:
            dataSet: Dataset with new feature generated.
        """
        col_name = 'MFI_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        df = df2.copy()
        
        print('column name: ', col_name)
        df['Up_or_Down'] = 0
        df.loc[(df['Close'] > df['Close'].shift(-1)), 'Up_or_Down'] = 1
        df.loc[(df['Close'] < df['Close'].shift(-1)), 'Up_or_Down'] = 2
        #print(df['Up_or_Down'].head(10))

        # 1 typical price
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        #print('====tp====')
        #print(tp.head(10))
        # 2 money flow
        mf = tp * df['Volume']
        #print('====mf====')
        #print(mf.head(10))
        # 3 positive and negative money flow with n periods
        df['1p_Positive_Money_Flow'] = 0.0
        df.loc[df['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
        n_positive_mf = df['1p_Positive_Money_Flow'].rolling(period).sum()

        df['1p_Negative_Money_Flow'] = 0.0
        df.loc[df['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
        n_negative_mf = df['1p_Negative_Money_Flow'].rolling(period).sum()
        #df.drop(['1p_Positive_Money_Flow','Up_or_Down','1p_Negative_Money_Flow'], axis=1)  
        
        
        # 4 money flow index
        mr = n_positive_mf / n_negative_mf
        mr = (100 - (100 / (1 + mr)))
        #print('============mr==========')
        #print(mr.tail(20))
        df[col_name] = (100 - (100 / (1 + n_positive_mf / n_negative_mf)))
        #df = df.drop(['1p_Positive_Money_Flow','Up_or_Down','1p_Negative_Money_Flow'], axis='columns')
        
        df2[col_name] = df[col_name]
        del df
        #print('============df[col_name]==========')
        #print(col_name)
        print(list(df2))
        
        return df2
    
 
class CustVolumeStudies:
    """Group of Custom Volume studies"""
    def ease_OfMvmnt(self, data, period):
        """Ease of Movement (EMV) is a volume-based oscillator which
        was developed by Richard Arms. EVM indicates the ease with which
        the prices rise or fall taking into account the volume of the
        security.
           Args:
                data: full dataframe
                volume: volume of issue
                feature_dict: Dictionary of added features
           Return:
                dataframe with EVM added
                feature_dict
        """
        col_name = 'EMV_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
        br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
        EVM = dm / br 
        EVM_MA = EVM.rolling(window=period, center=False).mean() 
        data[col_name] = EVM_MA
        return data
    
if __name__ == "__main__":
    from Code.lib.plot_utils import PlotUtility
    from Code.lib.retrieve_data import DataRetrieve
    from Code.lib.feature_generator import FeatureGenerator
    from Code.lib.transformers import Transformers
    
    taLibVolSt = TALibVolumeStudies()
    custVolSt = CustVolumeStudies()
    plotIt = PlotUtility()
    dSet = DataRetrieve()
    transf = Transformers()
    featureGen = FeatureGenerator()
    
    issue = "TLT"

    df = dSet.read_issue_data(issue)
    lastRow = df.shape[0]
    dataLoadEndDate = df.Date[lastRow-100]
    
    dataLoadStartDate = df.Date[lastRow-3000]
    dataSet = dSet.set_date_range(df,
                                  dataLoadStartDate,
                                  dataLoadEndDate
                                  )      
    dataSet.fillna(method='ffill', inplace=True)
    input_dict = {} # initialize 
    input_dict = input_dict = {'f1': 
                      {'fname' : 'PPO', 
                       'params' : [2,5],
                       'transform' : ['Normalized', 20]
                       },
                      'f2': 
                      {'fname' : 'MFI', 
                       'params' : [2],
                       'transform' : ['Normalized', 20]
                       },
                      'f3': 
                      {'fname' : 'MFI', 
                       'params' : [2],
                       'transform' : ['Scaler', 'Robust']
                       },
                      'f4': 
                      {'fname' : 'OBV', 
                       'params' : [],
                       'transform' : ['Normalized', 20]
                       }
                     }
    
    dataSet2 = featureGen.generate_features(dataSet, input_dict)
#    dataSet = taLibVolSt.ChaikinAD(dataSet)
#    dataSet = taLibVolSt.ChaikinADOSC(dataSet, 3, 10)
#    dataSet = taLibVolSt.OBV(dataSet)
#    dataSet = taLibVolSt.MFI(dataSet, 14)
#    dataSet = custVolSt.ease_OfMvmnt(dataSet, 14)
    
    startDate = "2015-02-01"
    endDate = "2015-06-12"
    plotDF = dataSet2[startDate:endDate]
    
    # Set up dictionary and plot HigherClose
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
