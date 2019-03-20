# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:04:07 2019

@author: kruegkj
"""

import random
import os
import datetime

from Code.lib.retrieve_data import DataRetrieve

current_directory = os.getcwd()

dSet = DataRetrieve()

def gen_dict_name():
    # generate random input_dict name
    a = random.sample(range(10),3)
    sys_date = datetime.datetime.now().strftime("%Y%m%d")
    name = "input_dict_" + str(sys_date) + "_" + str(a[0]) + str(a[1]) + str(a[2])
    return name

feature_func_dict = {'mom_function': {
                        'RSI':{'num_params': 1, 'param_range1': [2,10]},
                        'PPO':{'num_params': 2, 'param_range1': [2,10], 'param_range2': [4,20]},
                        'CMO':{'num_params': 1, 'param_range1': [2,10]},
                        'CCI':{'num_params': 1, 'param_range1': [2,10]},
                        'ROC':{'num_params': 1, 'param_range1': [2,10]},
                        'UltimateOscillator': {'num_params': 3, 'param_range1': [1,10], 'param_range2': [2,20], 'param_range3': [3,30]}
                         },
                 
                     'volume_function': {
                        'ChaikinAD':{'num_params': 0},
                        'ChaikinADOSC':{'num_params': 2,
                                        'param_range1': [3,10],
                                        'param_range2': [5,12]},
                        'OBV':{'num_params': 0},
                        'MFI':{'num_params': 1, 'param_range1': [1,10]},
                        'ease_OfMvmnt':{'num_params': 1, 'param_range1': [1,10]}
                         },
                     'transform_function': {
                         'Lag':{'num_params': 1, 'param_range1': [1,10]}
                         },
                     'transform2_function': {
                         'Lag':{'num_params': 1, 'param_range1': [3,10]}
                         },
                     'osc_Studies_function': {
                         'DetrendPO': {'num_params': 1, 'param_range1': [2,10]}
                         },
                     'volatility_function': {
                         'ATR': {'num_params': 1, 'param_range1': [1,10]},
                         'NATR': {'num_params': 1, 'param_range1': [1,10]},
                         'ATRRatio': {'num_params': 2, 'param_range1': [1,10], 'param_range2': [2,15]},
                         'DeltaATRRatio': {'num_params': 2,
                                           'param_range1': [2,10],
                                           'param_range2': [3,15]},
                         'BBWidth': {'num_params': 1, 'param_range1': [2,10]}
                         },
#                     'candle_functions': {
#                         'HigherClose': {'num_params': 1, 'param_range1': [1,10]},
#                         'LowerClose': {'num_params': 1, 'param_range1': [1,10]}
#                         },
                     'overlap_ma1_functions': {
                         'exp_MA': {'num_params': 1, 'param_range1': [2,10]},
                         'simple_MA': {'num_params': 1, 'param_range1': [2,10]},
                         'weighted_MA': {'num_params': 1, 'param_range1': [2,10]},
                         'triple_EMA': {'num_params': 1, 'param_range1': [2,10]}
                         },
                     'overlap_ma2_functions': {
                         'triangMA': {'num_params': 1, 'param_range1': [2,10]},
                         'dblEMA': {'num_params': 1, 'param_range1': [2,10]},
                         'kaufman_AMA': {'num_params': 1, 'param_range1': [2,10]},
                         'delta_MESA_AMA': {'num_params': 0}
                         },
                      'overlap_price_functions': {
                         'inst_Trendline': {'num_params': 0},
                         'mid_point': {'num_params': 1, 'param_range1': [2,10]},
                         'mid_price': {'num_params': 1, 'param_range1': [2,10]},
                         'pSAR': {'num_params': 1, 'param_range1': [2,10]}
                         }
                     }

transform_list = [10, 20, 30, 40, 50]

# create  
list_of_input_dicts = []

num_dicts_to_make = 40
dict_ctr = 0

while dict_ctr < num_dicts_to_make:
    dict_name = gen_dict_name()
    list_of_input_dicts.append(dict_name)

    temp_dict = {}
    # TO-DO: Create list of functions to use, select random function from
    # each category list
    i = 0
    for k in feature_func_dict:
        i += 1
        #print(k)
    
        keys = list(feature_func_dict[k])
        #print(keys)
    
        params_value = []
        var = 'f' + str(i)
    #    #TO-DO: select key for category selected
        curr_key = random.choice(keys)
        #print("\n",curr_key)
        num_params = feature_func_dict[k][curr_key]['num_params']
        #print("NUM params from dict: ", num_params)
        if num_params == 0:
            #print("no params")
            params_value = []
        else:
            for ll in range(1,num_params+1):  
                p_value = feature_func_dict[k][curr_key]['param_range'+str(ll)]
                
                if ll > 1:
                    #print("more than one param")
                    #print(params_value[ll-2])
                    # set min of range at least one higher than previous val
                    random_range = random.randrange(params_value[ll-2]+1, p_value[1])
                else:
                    #print("1st param")
                    random_range = random.randrange(p_value[0], p_value[1])
                params_value.append(random_range)
                #print("--> ",params_value)
        temp = {}
        temp['fname'] = curr_key
        temp['params'] = params_value
        # hard coded for now for MFI mormalization problem, 
        # fix later
        if curr_key == 'MFI':
            temp['transform'] = ['Scaler', 'Robust']
        else:
            temp['transform'] = ['Normalized', random.choice(transform_list)]
        #print(temp)
        temp_dict[var]=temp
        
        #temp_dict[var].append(temp)
    #print(temp_dict)
    #print("Saving ", dict_name)
    dSet.save_pickle(temp_dict, current_directory, dict_name)
    dict_ctr += 1

program_name = "IS_Day_by_day_TLT_v1.py"

for dict_name in list_of_input_dicts:
    print(dict_name)
    current_directory = os.getcwd()
        # add try/catch block later
    filename = dict_name
    input_dict = dSet.load_pickle(current_directory, filename)
    print(input_dict)
    
    print('python {0} {1} "{2}"'.format(program_name,'""',dict_name))
    #print(list_of_input_dicts[i])
    os.system('python {0} {1} "{2}"'.format(program_name,'""',dict_name))
