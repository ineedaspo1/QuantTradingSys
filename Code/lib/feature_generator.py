# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:24:33 2018
@author: KRUEGKJ
feature_generator.py
"""
import numpy as np
from Code.lib.retrieve_data import DataRetrieve
import functools
import pickle
from Code.lib.config import current_feature, feature_dict


class FeatureGenerator:
    def get_from_dict(self, dataDict, mapList):
        """Iterate nested dictionary"""
        try:
            return functools.reduce(dict.get, mapList, dataDict)
        except TypeError:
            return None  # or some other default value
        
    def generate_features(self, df, input_dict):
        funcDict = DataRetrieve.load_obj(self, 'func_dict')
        for key in input_dict.keys():
            print(key)
            path = [key, 'fname']
            #print('fname: ', self.get_from_dict(input_dict, path))
            func_name = self.get_from_dict(input_dict, path)
            
            path = [key, 'params']
            #print('params: ', self.get_from_dict(input_dict, path))
            params = self.get_from_dict(input_dict, path)  
            df = funcDict[func_name](df, *params)
            print("Current feature: ", current_feature['Latest'])
            
            path = [key, 'transform']
            print('transform: ', self.get_from_dict(input_dict, path))
            do_transform = self.get_from_dict(input_dict, path)
            
            if do_transform:
                #print('!!!!', do_transform[0], )
                pass_params = (do_transform[1::])
                #print("pass params" , pass_params)
                #print("Current feature: ", current_feature['Latest'])
                df = funcDict[do_transform[0]](df,
                                               current_feature['Latest'],
                                               *pass_params
                                               )
                #print("Current feature: ", current_feature['Latest'])
        return df
    
if __name__ == "__main__":
    from Code.lib.plot_utils import PlotUtility as plotIt
    from Code.lib.ta_momentum_studies import TALibMomentumStudies
    from Code.lib.ta_volume_studies import TALibVolumeStudies, CustVolumeStudies
    from Code.lib.ta_volatility_studies import TALibVolatilityStudies
    from Code.lib.ta_overlap_studies import TALibOverlapStudies
    from Code.lib.transformers import Transformers
    from Code.lib.oscillator_studies import OscialltorStudies
    from Code.lib.candle_indicators import CandleIndicators
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    
    taLibVolSt = TALibVolumeStudies()
    taLibMomSt = TALibMomentumStudies()
    transf = Transformers()
    oscSt = OscialltorStudies()
    vStud = TALibVolatilityStudies()
    feat_gen = FeatureGenerator()
    candle_ind = CandleIndicators()
    taLibVolSt = TALibVolumeStudies()
    custVolSt = CustVolumeStudies()
    taLibOS = TALibOverlapStudies()
    featureGen = FeatureGenerator()
    #plotIt = PlotUtility()
    
    # move this dict to be read from file
#    functionDict = {
#            "RSI"               : taLibMomSt.RSI,
#            "PPO"               : taLibMomSt.PPO,
#            "CMO"               : taLibMomSt.CMO,
#            "CCI"               : taLibMomSt.CCI,
#            "ROC"               : taLibMomSt.rate_OfChg,
#            "UltimateOscillator": taLibMomSt.UltOsc,
#            "Normalized"        : transf.normalizer,
#            "Zscore"            : transf.zScore,
#            "Scaler"            : transf.scaler,
#            "Center"            : transf.centering,
#            "Lag"               : transf.add_lag,
#            "DetrendPO"         : oscSt.detrend_PO,
#            "ATR"               : vStud.ATR,
#            "NATR"              : vStud.NATR,
#            "ATRRatio"          : vStud.ATR_Ratio,
#            "DeltaATRRatio"     : vStud.delta_ATR_Ratio,
#            "BBWidth"           : vStud.BBWidth,
#            "HigherClose"       : candle_ind.higher_close,
#            "LowerClose"        : candle_ind.lower_close,
#            "ChaikinAD"         : taLibVolSt.ChaikinAD,
#            "ChaikinADOSC"      : taLibVolSt.ChaikinADOSC,
#            "OBV"               : taLibVolSt.OBV,
#            "MFI"               : taLibVolSt.MFI,
#            "ease_OfMvmnt"      : custVolSt.ease_OfMvmnt,
#            "exp_MA"            : taLibOS.exp_MA,
#            "simple_MA"         : taLibOS.simple_MA,
#            "weighted_MA"       : taLibOS.weighted_MA,
#            "triple_EMA"        : taLibOS.triple_EMA,
#            "triangMA"          : taLibOS.triangMA,
#            "dblEMA"            : taLibOS.dblEMA,
#            "kaufman_AMA"       : taLibOS.kaufman_AMA,
#            "delta_MESA_AMA"    : taLibOS.delta_MESA_AMA,
#            "inst_Trendline"    : taLibOS.inst_Trendline,
#            "mid_point"         : taLibOS.mid_point,
#            "mid_price"         : taLibOS.mid_price,
#            "pSAR"              : taLibOS.pSAR
#            }
        
    dSet = DataRetrieve()
#    dSet.save_obj(functionDict, 'func_dict')
    #funcDict = dSet.load_obj('func_dict')
    
    df = dSet.read_issue_data(issue)
    df = dSet.set_date_range(df, dataLoadStartDate,dataLoadEndDate)
    
    # Example
    # RSI, 20d period, no transform
    input_dict = {} # initialize 
    input_dict = {'f1': 
                  {'fname' : 'RSI', 
                   'params' : [10]
                   },
                  'f2': 
                  {'fname' : 'UltimateOscillator', 
                   'params' : [10 , 20, 30]
                   },
                  'f3': 
                  {'fname' : 'UltimateOscillator',
                   'params' : [],
                   'transform' : ['Normalized', 100]
                   },
                  'f4': 
                  {'fname' : 'RSI', 
                   'params' : [10],
                   'transform' : ['Zscore', 3]
                   },
                  'f5': 
                  {'fname' : 'RSI', 
                   'params' : [3],
                   'transform' : ['Scaler', 'robust']
                   },
                  'f6': 
                  {'fname' : 'RSI', 
                   'params' : [10],
                   'transform' : ['Center', 3]
                   },
                  'f7': 
                  {'fname' : 'Lag', 
                   'params' : ['Close', 3]
                   },
                  'f8': 
                  {'fname' : 'PPO', 
                   'params' : [12, 26]
                   },
                  'f9': 
                  {'fname' : 'CMO', 
                   'params' : [10]
                   },
                  'f10': 
                  {'fname' : 'CCI', 
                   'params' : [10]
                   },
                  'f11': 
                  {'fname' : 'ROC', 
                   'params' : [10]
                   },
                  'f12': 
                  {'fname' : 'ATR', 
                   'params' : [10]
                   },
                  'f13': 
                  {'fname' : 'NATR', 
                   'params' : [10]
                   },
                  'f14': 
                  {'fname' : 'ATRRatio', 
                   'params' : [10, 30]
                   },
                  'f15': 
                  {'fname' : 'DeltaATRRatio', 
                   'params' : [10, 50]
                   },
                  'f16': 
                  {'fname' : 'BBWidth', 
                   'params' : [10]
                   },
                  'f17': 
                  {'fname' : 'HigherClose', 
                   'params' : [4]
                   },
                  'f18': 
                  {'fname' : 'LowerClose', 
                   'params' : [4]
                   },
                  'f19': 
                  {'fname' : 'ChaikinAD', 
                   'params' : []
                   },
                  'f20': 
                  {'fname' : 'ChaikinADOSC', 
                   'params' : [4, 10],
                   'transform' : ['Normalized', 100]
                   },
                  'f21': 
                  {'fname' : 'OBV', 
                   'params' : [],
                   'transform' : ['Zscore', 3]
                   },
                  'f22': 
                  {'fname' : 'MFI', 
                   'params' : [14],
                   'transform' : ['Zscore', 3]
                   },
                  'f23': 
                  {'fname' : 'ease_OfMvmnt', 
                   'params' : [14],
                   'transform' : ['Zscore', 3]
                   },
                  'f24': 
                  {'fname' : 'exp_MA', 
                   'params' : [4]
                   },
                  'f25': 
                  {'fname' : 'simple_MA', 
                   'params' : [4]
                   },
                  'f26': 
                  {'fname' : 'weighted_MA', 
                   'params' : [4]
                   },
                  'f27': 
                  {'fname' : 'triple_EMA', 
                   'params' : [4]
                   },
                  'f28': 
                  {'fname' : 'triangMA', 
                   'params' : [4]
                   },
                  'f29': 
                  {'fname' : 'dblEMA', 
                   'params' : [4]
                   },
                  'f30': 
                  {'fname' : 'kaufman_AMA', 
                   'params' : [4]
                   },
                  'f31': 
                  {'fname' : 'delta_MESA_AMA', 
                   'params' : [0.9, 0.1],
                   'transform' : ['Normalized', 20]
                   },
                  'f32': 
                  {'fname' : 'inst_Trendline', 
                   'params' : []
                   },
                  'f33': 
                  {'fname' : 'mid_point', 
                   'params' : [4]
                   },
                  'f34': 
                  {'fname' : 'mid_price', 
                   'params' : [4]
                   },
                  'f35': 
                  {'fname' : 'pSAR', 
                   'params' : [4]
                   }
                 }
               
    df = featureGen.generate_features(df, input_dict)
            
    # Plot price and indicators
    startDate = "2015-02-01"
    endDate = "2015-06-30"

    plotDataSet = df[startDate:endDate]
#    plot_dict = {}
#    plot_dict['Issue'] = issue
#    plot_dict['Plot_Vars'] = list(feature_dict.keys())
#    plot_dict['Volume'] = 'Yes'
#    plotIt.price_Ind_Vol_Plot(plot_dict, plotDataSet)
    
    # Drop columns where value is 'Drop'
    col_vals=['Open', 'High', 'Close', 'Low']
    corrDataSet = dSet.drop_columns(plotDataSet, col_vals)
    col_vals = [k for k,v in feature_dict.items() if v == 'Drop']
    corrDataSet = dSet.drop_columns(corrDataSet, col_vals)
    plotIt.correlation_matrix(corrDataSet)
