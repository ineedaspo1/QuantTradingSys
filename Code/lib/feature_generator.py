# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:24:33 2018
@author: KRUEGKJ
feature_generator.py
"""
import numpy as np
#from config import *
import functools


class FeatureGenerator:
    pass

if __name__ == "__main__":
    from plot_utils import *
    from retrieve_data import *
    from ta_momentum_studies import *
    from ta_volume_studies import *
    from ta_volume_studies import *
    from ta_volatility_studies import *
    from transformers import *
    from oscillator_studies import *
    from config import current_feature, feature_dict
    
    dataLoadStartDate = "2014-04-01"
    dataLoadEndDate = "2018-04-01"
    issue = "TLT"
    
    taLibVolSt = TALibVolumeStudies()
    taLibMomSt = TALibMomentumStudies()
    transf = Transformers()
    oscSt = OscialltorStudies()
    vStud = TALibVolatilityStudies()
    feat_gen = FeatureGenerator()
    
    # move this dict to be read from file
    funcDict = {
            "RSI" : taLibMomSt.RSI,
            "PPO" : taLibMomSt.PPO,
            "CMO" : taLibMomSt.CMO,
            "CCI" : taLibMomSt.CCI,
            "ROC" : taLibMomSt.rate_OfChg,
            "UltimateOscillator": taLibMomSt.UltOsc,
            "Normalized": transf.normalizer,
            "Zscore" : transf.zScore,
            "Scaler" : transf.scaler,
            "Center" : transf.centering,
            "Lag" : transf.add_lag,
            "DetrendPO" : oscSt.detrend_PO,
            "ATR" : vStud.ATR,
            "NATR" : vStud.NATR,
            "ATRRatio" : vStud.ATR_Ratio,
            "DeltaATRRatio" : vStud.delta_ATR_Ratio,
            "BBWidth" : vStud.BBWidth
            }
    
    def get_from_dict(dataDict, mapList):
        """Iterate nested dictionary"""
        try:
            return functools.reduce(dict.get, mapList, dataDict)
        except TypeError:
            return None  # or some other default value

    dSet = DataRetrieve()
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
                   }
                 }
                  
    for key in input_dict.keys():
        #print(key)
        path = [key, 'fname']
        print('fname: ', get_from_dict(input_dict, path))
        func_name = get_from_dict(input_dict, path)
        
        path = [key, 'params']
        print('params: ', get_from_dict(input_dict, path))
        params = get_from_dict(input_dict, path)  
        df = funcDict[func_name](df, *params)
        print("Current feature: ", current_feature['Latest'])
        
        path = [key, 'transform']
        #print('transform: ', get_from_dict(input_dict, path), '\n')
        do_transform = get_from_dict(input_dict, path)
        
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
            
    # Plot price and indicators
    startDate = "2015-02-01"
    endDate = "2015-06-30"

    plotDataSet = df[startDate:endDate]
    plot_dict = {}
    plot_dict['Issue'] = issue
    plot_dict['Plot_Vars'] = list(feature_dict.keys())
    plot_dict['Volume'] = 'Yes'
    plotIt.price_Ind_Vol_Plot(plot_dict, plotDataSet)