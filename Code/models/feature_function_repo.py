# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:58:16 2019

@author: kruegkj

 Input_dict = {'f<num>':
			     {'fname': '<function name from feature generator list'>,
			      'params': [<params>],
			      'transform': ['<type>', <param>]
			      }
              }
"""

# TO-DO: Create multiple category based functions
feature_func_dict = {'mom_function': {
                        'RSI':{'num_params': 1, 'param_range1': [1,10]},
                        'PPO':{'num_params': 2, 'param_range1': [1,10], 'param_range2': [5,20]},
                        'CMO':{'num_params': 1, 'param_range1': [1,10]},
                        'CCI':{'num_params': 1, 'param_range1': [2,10]},
                        'ROC':{'num_params': 1, 'param_range1': [1,10]},
                        'UltimateOscillator': {'num_params': 3, 'param_range1': [1,10], 'param_range2': [11,20], 'param_range3': [20,30]}
                         },
                 
                     'volume_function': {
                        'ChaikinAD':{'num_params': 0},
                        'ChaikinADOSC':{'num_params': 2, 'param_range1': [1,10], 'param_range2': [1,10]},
                        'OBV':{'num_params': 0},
                        'MFI':{'num_params': 1, 'param_range1': [1,10]},
                        'ease_OfMvmnt':{'num_params': 1, 'param_range1': [1,10]}
                         },
                     'transform_function': {
                         'Lag':{'num_params': 1, 'param_range1': [1,10]}
                         },
                     'osc_Studies_function': {
                         'DetrendPO': {'num_params': 1, 'param_range1': [1,10]}
                         },
                     'volatility_function': {
                         'ATR': {'num_params': 1, 'param_range1': [1,10]},
                         'NATR': {'num_params': 1, 'param_range1': [1,10]},
                         'ATRRatio': {'num_params': 2, 'param_range1': [1,10], 'param_range2': [5,15]},
                         'DeltaATRRatio': {'num_params': 2, 'param_range1': [1,10], 'param_range2': [5,15]},
                         'BBWidth': {'num_params': 1, 'param_range1': [1,10]}
                         },
                     'candle_functions': {
                         'HigherClose': {'num_params': 1, 'param_range1': [1,10]},
                         'LowerClose': {'num_params': 1, 'param_range1': [1,10]}
                         },
                     'overlap_ma_functions': {
                         'exp_MA': {'num_params': 1, 'param_range1': [1,10]},
                         'simple_MA': {'num_params': 1, 'param_range1': [1,10]},
                         'weighted_MA': {'num_params': 1, 'param_range1': [1,10]},
                         'triple_EMA': {'num_params': 1, 'param_range1': [1,10]},
                         'triangMA': {'num_params': 1, 'param_range1': [1,10]},
                         'dblEMA': {'num_params': 1, 'param_range1': [1,10]},
                         'kaufman_AMA': {'num_params': 1, 'param_range1': [1,10]},
                         'delta_MESA_AMA': {'num_params': 2, 'param_range1': [0.1, 0.9], 'param_range2': [0.05, 0.2]},
                         },
                      'overlap_price_functions': {
                         'inst_Trendline': {'num_params': 0},
                         'mid_point': {'num_params': 1, 'param_range1': [1,10]},
                         'mid_price': {'num_params': 1, 'param_range1': [1,10]},
                         'pSAR': {'num_params': 1, 'param_range1': [1,10]}
                         }
                     }









#                
#
#                "exp_MA"            : taLibOS.exp_MA,
#                "simple_MA"         : taLibOS.simple_MA,
#                "weighted_MA"       : taLibOS.weighted_MA,
#                "triple_EMA"        : taLibOS.triple_EMA,
#                "triangMA"          : taLibOS.triangMA,
#                "dblEMA"            : taLibOS.dblEMA,
#                "kaufman_AMA"       : taLibOS.kaufman_AMA,
#                "delta_MESA_AMA"    : taLibOS.delta_MESA_AMA,
                     
#                "inst_Trendline"    : taLibOS.inst_Trendline,
#                "mid_point"         : taLibOS.mid_point,
#                "mid_price"         : taLibOS.mid_price,
#                "pSAR"              : taLibOS.pSAR