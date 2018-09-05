# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018

@author: KRUEGKJ

ta_volume_studies.py
"""
import sys
sys.path.append('../lib')
sys.path.append('../utilities')

import talib as ta


class TALibVolumeStudies:
    """Group of Volume studies utilized fromTALib """
    def ChaikinAD(self, high, low, close, volume, feature_dict):
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
        feature_dict['ChaikinAD']='Keep'
        chaikinAD = ta.AD(high,
            low,
            close, 
            volume
            )
        return chaikinAD, feature_dict
    
    def ChaikinADOSC(self, high, low, close, volume, fastperiod, slowperiod, feature_dict):
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
        feature_dict['ChaikinADOSC_f'+str(fastperiod)+'_s'+str(slowperiod)]='Keep'
        chaikinADOSC = ta.ADOSC(high,
            low,
            close, 
            volume,
            fastperiod,
            slowperiod
            )
        return chaikinADOSC, feature_dict
    
    def OBV(self, close, volume, feature_dict):
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
        feature_dict['OBV']='Keep'
        onBalVol = ta.OBV(close, 
            volume
            )
        return onBalVol, feature_dict
    
    def MFI(self, dataSet, period, feature_dict):
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
        feature_dict['MFI_'+str(period)]='Keep'
        dataSet['Up_or_Down'] = 0
        dataSet.loc[(dataSet['Close'] > dataSet['Close'].shift(1)), 'Up_or_Down'] = 1
        dataSet.loc[(dataSet['Close'] < dataSet['Close'].shift(1)), 'Up_or_Down'] = 2

        # 1 typical price
        tp = (dataSet['High'] + dataSet['Low'] + dataSet['Close']) / 3.
        # 2 money flow
        mf = tp * dataSet['Volume']
        # 3 positive and negative money flow with n periods
        dataSet['1p_Positive_Money_Flow'] = 0.0
        dataSet.loc[dataSet['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
        n_positive_mf = dataSet['1p_Positive_Money_Flow'].rolling(period).sum()

        dataSet['1p_Negative_Money_Flow'] = 0.0
        dataSet.loc[dataSet['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
        n_negative_mf = dataSet['1p_Negative_Money_Flow'].rolling(period).sum()
        dataSet = dataSet.drop(['1p_Positive_Money_Flow','Up_or_Down','1p_Negative_Money_Flow'], axis=1)  
        
        # 4 money flow index
        mr = n_positive_mf / n_negative_mf
        mr = (100 - (100 / (1 + mr)))
        dataSet['MFI'] = mr
        
        return dataSet, feature_dict    
 
class CustVolumeStudies:
    """Group of Custom Volume studies"""
    def ease_OfMvmnt(self, data, period, feature_dict):
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
        feature_dict['EMV_'+str(period)]='Keep'
        dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
        br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
        EVM = dm / br 
        EVM_MA = EVM.rolling(window=period, center=False).mean() 
        data['EVM'] = EVM_MA
        return data, feature_dict
    
if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    
    taLibVolSt = TALibVolumeStudies()
    custVolSt = CustVolumeStudies()
    plotIt = PlotUtility()
    dSet = DataRetrieve()
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    feature_dict = {}
    
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet, dataLoadStartDate,dataLoadEndDate)
        
    dataSet['ChaikinAD'], feature_dict = taLibVolSt.ChaikinAD(dataSet.High.values, dataSet.Low.values, dataSet.Pri.values, dataSet.Volume, feature_dict)
    dataSet['ChaikinADOSC'], feature_dict = taLibVolSt.ChaikinADOSC(dataSet.High.values, dataSet.Low.values, dataSet.Pri.values, dataSet.Volume, 3, 10, feature_dict)
    dataSet['OBV'], feature_dict = taLibVolSt.OBV(dataSet.Pri.values, dataSet.Volume, feature_dict)
    # MFI
    dataSet, feature_dict = taLibVolSt.MFI(dataSet, 14, feature_dict)
    dataSet, feature_dict = custVolSt.ease_OfMvmnt(dataSet, 14, feature_dict)
    
    startDate = "2015-02-01"
    endDate = "2015-04-30"
    plotDF = dataSet[startDate:endDate]
    
    # Set up dictionary and plot HigherClose
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = ['ChaikinAD', 'ChaikinADOSC', 'OBV', 'MFI', 'EVM']
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
