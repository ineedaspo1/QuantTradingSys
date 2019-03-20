# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:00 2018
@author: KRUEGKJ
ta_overlap_studies.py
"""
import talib as ta
import numpy as np
import matplotlib.pyplot as plt
from Code.lib.config import current_feature, feature_dict

class TALibOverlapStudies:
    """Group of Momentum studies utilized fromTALib """

    def boll_bands(self, df, period, nbdev=2):
        """Bollinger Bands
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                nbdev = Number of std dev's
                feature_dict: Dictionary of added features
            Return:
                Upper, middle, lower prices of Bollinger Band
                feature_dict
        """
        col_name = 'BollBands' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        
        stddev = nbdev
        df['BBUB'], df['BBMB'], df['BBLB'] = ta.BBANDS(
                df.Close,
                timeperiod=period,
                nbdevup=stddev,
                nbdevdn=stddev,
                matype=0  # Moving average type: 0-simple
                )
        return df

    def exp_MA(self, df, period):
        """Exponential MA
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                expMA
                feature_dict
        """
        col_name = 'EMA_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.EMA(df.Close, period)
        return df

    def simple_MA(self, df, period):
        """Simple MA
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                simpleMA
                feature_dict
        """
        col_name = 'SMA_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.SMA(df.Close, period)
        return df

    def weighted_MA(self, df, period):
        """Weighted MA calculates a weight for each value in the series.
        The more recent values are assigned greater weights.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                weightedMA
                feature_dict
        """
        col_name = 'WghtdMA_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
    
        df[col_name] = ta.WMA(df.Close, period)
        return df

    def triple_EMA(self, df, period):
        """Triple MA  a smoothing indicator with less lag than a straight
        exponential moving average.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                tripleEMA
                feature_dict
        """
        col_name = 'TripleEMA_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        df[col_name] = ta.TEMA(df.Close, period)
        return df

    def triangMA(self, df, period):
        """The Triangular Moving Average is a form of Weighted Moving
        Average wherein the weights are assigned in a triangular pattern.
        For example, the weights for a 7 period Triangular Moving Average
        would be 1, 2, 3, 4, 3, 2, 1. This gives more weight to the middle
        of the time series and less weight to the oldest and newest data.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                triangularMA
                feature_dict
        """
        col_name = 'TriangMA_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        df[col_name] = ta.TRIMA(df.Close, period)
        return df

    def dblEMA(self, df, period):
        """The DEMA is a smoothing indicator with less lag than a straight
        exponential moving average
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                dblExpMA
                feature_dict
        """
        col_name = 'DblEMA_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'

        df[col_name] = ta.DEMA(df.Close, period)
        return df

    def kaufman_AMA(self, df, period):
        """The KAMA is a moving average designed to account for market noise
        or volatility. KAMA will closely follow prices when the price swings
        are relatively small and the noise is low. KAMA will adjust when the
        price swings widen and follow prices from a greater distance. This
        trend-following indicator can be used to identify the overall trend,
        time turning points and filter price movements.
            Args:
                close: Closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                kaufmanAMA
                feature_dict
            To Update: KAMA has 3 parameteres, but TA-Lib only exposes 1. Move
                to custom code to expose all 3
        """
        col_name = 'KAMA_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.KAMA(df.Close, period)
        return df

    def mesa_AMA(self, df, flimit=0.5, slimit=0.05):
        """The MESA Adaptive Moving Average is a technical trend-following
        indicator which, according to its creator, adapts to price movement
        “based on the rate change of phase as measured by the Hilbert Transform
        Discriminator”.
        http://www.binarytribune.com/forex-trading-indicators/ehlers-mesa-adaptive-moving-average
            Args:
                close: Closing price of instrument
                flimit: number of fast time periods in the calculation
                slimit: number of slow time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                mama: MESA Adaptive MA
                fama: Following AMA being applied to mama
                feature_dict
        """
        col_name = 'MesaAMA_f' + str(flimit) + '_s' + str(slimit)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df['MAMA'], df['FAMA'] = ta.MAMA(df.Close, flimit, slimit)
        return df
    
    def delta_MESA_AMA(self, df, flimit=0.5, slimit=0.05):
        col_name = 'DeltaMesaAMA_f' + str(flimit) + '_s' + str(slimit)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df['MAMA'], df['FAMA'] = ta.MAMA(df.Close, flimit, slimit)
        df[col_name] = df['MAMA'] - df['FAMA']
        feature_dict['MAMA'] = 'Drop'
        feature_dict['FAMA'] = 'Drop'
        return df

    def inst_Trendline(self, df):
        """The Ehlers Hilbert Transform - Instantaneous Trendline is a
         smoothed trendline,
            Args:
                close: Closing price of instrument
                feature_dict: Dictionary of added features
            Return:
                instTrendline: resulting signal
                feature_dict
        """
        col_name = 'InstTrendline'
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.HT_TRENDLINE(df.Close)
        return df

    def mid_point(self, df, period):
        """(highest value + lowest value)/2 over period
            Args:
                close: closing price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                midPrice: resulting signal
                feature_dict
        """
        col_name = 'Midpoint_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.MIDPOINT(df.Close, period)
        return df

    def mid_price(self, df, period):
        """(highest high + lowest low)/2 over period
            Args:
                high, low: high, low price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                midPrice: resulting signal
                feature_dict
        """
        col_name = 'Midprice_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.MIDPRICE(df.High, df.Low, period)
        return df

    def pSAR(self, df, period):
        """The parabolic SAR is a technical indicator used to determine the
        price direction of an asset, as well draw attention to when the price
        direction is changing.
            Args:
                high, low: high, low price of instrument
                period: number of time periods in the calculation
                feature_dict: Dictionary of added features
            Return:
                pSAR: resulting signal
                feature_dict
        """
        col_name = 'PSAR_' + str(period)
        current_feature['Latest'] = col_name
        feature_dict[col_name] = 'Keep'
        df[col_name] = ta.SAR(df.High, df.Low, period)
        return df

if __name__ == "__main__":   
    from Code.lib.plot_utils import PlotUtility
    from Code.lib.retrieve_data import DataRetrieve
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    #feature_dict = {}

    taLibOS = TALibOverlapStudies()

    dSet = DataRetrieve()
    dataSet = dSet.read_issue_data(issue)
    dataSet = dSet.set_date_range(dataSet,
                                  dataLoadStartDate,
                                  dataLoadEndDate
                                  )
    
    dataSet = taLibOS.boll_bands(dataSet, 20, 2)
    dataSet = taLibOS.dblEMA(dataSet, 20)
    dataSet = taLibOS.kaufman_AMA(dataSet, 20)

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    
#    plot_dict = {}
#    plot_dict['Issue'] = issue
#    plot_dict['Plot_Vars'] = list(feature_dict.keys())
#    plot_dict['Volume'] = 'Yes'
#    plotIt.price_Ind_Vol_Plot(plot_dict, plotDF)
    
    N = len(plotDF)
    ind = np.arange(N)  # the evenly spaced plot indices

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return plotDF.Date[thisind].strftime('%Y-%m-%d')
    
    plt.figure(figsize=(15, 8))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    #  top.plot(ind, plotDF['Pri']) #
    #  bottom.bar(ind, plotDF['Volume'])
    top.plot(ind,
             plotDF['Close'],
             'k-', markersize=3,
             label=issue)
    top.plot(ind,
             plotDF['BBUB'], 'c-')
    top.plot(ind, plotDF['BBMB'], 'c--')
    top.plot(ind, plotDF['BBLB'], 'c-')
    top.plot(ind, plotDF['DblEMA_20'], 'y-')
    top.plot(ind, plotDF['KAMA_20'], 'g-')
    bottom.bar(ind, plotDF['Volume'], label='Volume')
    plt.subplots_adjust(hspace=0.05)
    #  set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    for ax in top, bottom:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()

        
    dataSet = taLibOS.mesa_AMA(dataSet, 0.9, 0.1)
    dataSet = taLibOS.delta_MESA_AMA(dataSet, 0.9, 0.1)
    dataSet = taLibOS.exp_MA(dataSet, 30)
    dataSet = taLibOS.inst_Trendline(dataSet)

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    N = len(plotDF)
    ind = np.arange(N)  # the evenly spaced plot indices
    plt.figure(figsize=(15, 8))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    #  top.plot(ind, plotDF['Pri']) #
    #  bottom.bar(ind, plotDF['Volume'])
    top.plot(ind,
             plotDF['Close'],
             'k-',
             markersize=3,
             label=issue)
    top.plot(ind, plotDF['MAMA'], 'g-')
    top.plot(ind, plotDF['FAMA'], 'r-')
    top.plot(ind, plotDF['EMA_30'], 'b--', alpha=0.6)
    top.plot(ind, plotDF['InstTrendline'], 'y-')
    bottom.bar(ind, plotDF['Volume'], label='Volume')
    plt.subplots_adjust(hspace=0.05)
    #  set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    for ax in top, bottom:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()

    dataSet = taLibOS.mid_point(dataSet, 30)
    dataSet = taLibOS.mid_price(dataSet,30)
    dataSet = taLibOS.pSAR(dataSet, 30)
    dataSet = taLibOS.simple_MA(dataSet, 30)
    dataSet = taLibOS.triple_EMA(dataSet, 30)
    dataSet = taLibOS.triangMA(dataSet, 30)
    dataSet = taLibOS.weighted_MA(dataSet, 30)

    startDate = "2015-02-01"
    endDate = "2015-06-30"
    plotDF = dataSet[startDate:endDate]
    N = len(plotDF)
    ind = np.arange(N)  # the evenly spaced plot indices
    plt.figure(figsize=(15, 8))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    #  top.plot(ind, plotDF['Pri']) #
    #  bottom.bar(ind, plotDF['Volume'])
    top.plot(ind,
             plotDF['Close'],
             'k-',
             markersize=3,
             label=issue
             )
    plot_list = list(feature_dict.keys())
    cnt = len(plot_list)
    for n in range(7,cnt):
        print (n)
        top.plot(ind, plotDF[plot_list[n]])
    bottom.bar(ind, plotDF['Volume'], label='Volume')
    plt.subplots_adjust(hspace=0.05)
    #  set the labels
    top.axes.get_xaxis().set_visible(True)
    top.set_title('TLT')
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    for ax in top, bottom:
        ax.label_outer()
        ax.legend(loc='upper left', frameon=True, fontsize=8)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.grid(True, which='both')
        ax.autoscale_view()
        ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
        ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        ax.minorticks_on()
