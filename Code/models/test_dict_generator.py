# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:30:17 2019

@author: KRUEGKJ
"""
import os
import datetime
import random
from Code.lib.retrieve_data import DataRetrieve

dSet = DataRetrieve()

current_directory = os.getcwd()

# create  
list_of_input_dicts = []

def gen_dict_name():
    # generate random input_dict name
    a = random.sample(range(10),3)
    sys_date = datetime.datetime.now().strftime("%Y%m%d")
    name = "input_dict_" + str(sys_date) + "_" + str(a[0]) + str(a[1]) + str(a[2])
    return name

test_input_dict = {} # initialize
test_input_dict = {'f1': 
                  {'fname' : 'PPO', 
                   'params' : [2,5],
                   'transform' : ['Normalized', 20]
                   },
                  'f2': 
                  {'fname' : 'RSI', 
                   'params' : [2],
                   'transform' : ['Normalized', 20]
                   },
                  'f3': 
                  {'fname' : 'RSI', 
                   'params' : [4],
                   'transform' : ['Normalized', 20]
                   },
                  'f4': 
                  {'fname' : 'PPO', 
                   'params' : [2,5],
                   'transform' : ['Normalized', 50]
                   },
                  'f5': 
                  {'fname' : 'RSI', 
                   'params' : [2],
                   'transform' : ['Normalized', 50]
                   },
                  'f6': 
                  {'fname' : 'RSI', 
                   'params' : [4],
                   'transform' : ['Normalized', 50]
                   }
                 }
dict_name = gen_dict_name()
list_of_input_dicts.append(dict_name)
print(list_of_input_dicts)
dSet.save_pickle(test_input_dict, current_directory, dict_name)

test_input_dict = {} # initialize
test_input_dict = {'f1': 
                  {'fname' : 'ATR', 
                   'params' : [5],
                   'transform' : ['Normalized', 20]
                   },
                  'f2': 
                  {'fname' : 'Lag', 
                   'params' : ['Close', 1],
                   'transform' : ['Normalized', 20]
                   },
                  'f3': 
                  {'fname' : 'Lag', 
                   'params' : ['Close', 2],
                   'transform' : ['Normalized', 20]
                   },
                  'f4': 
                  {'fname' : 'Lag', 
                   'params' : ['Close', 3],
                   'transform' : ['Normalized', 20]
                   },
                  'f5': 
                  {'fname' : 'Lag', 
                   'params' : ['Close', 4],
                   'transform' : ['Normalized', 20]
                   },
                  'f6': 
                  {'fname' : 'Lag', 
                   'params' : ['Close', 5],
                   'transform' : ['Normalized', 20]
                   }
                 }
dict_name = gen_dict_name()
list_of_input_dicts.append(dict_name)
print(list_of_input_dicts)
dSet.save_pickle(test_input_dict, current_directory, dict_name)

input_dict = {} # initialize 
input_dict = {'f1': {'fname': 'PPO', 'params': [2, 5], 'transform': ['Normalized', 20]},
  'f10': {'fname': 'kaufman_AMA',
  'params': [4],
  'transform': ['Normalized', 20]},
 'f2': {'fname': 'RSI', 'params': [2], 'transform': ['Normalized', 20]},
 'f3': {'fname': 'CMO', 'params': [5], 'transform': ['Normalized', 20]},
 'f4': {'fname': 'CCI', 'params': [10], 'transform': ['Normalized', 20]},
 'f5': {'fname': 'UltimateOscillator',
  'params': [10, 20, 30],
  'transform': ['Normalized', 20]},
 'f6': {'fname': 'ROC', 'params': [10], 'transform': ['Normalized', 20]},
 'f7': {'fname': 'Lag',
  'params': ['Close', 3],
  'transform': ['Normalized', 20]},
 'f8': {'fname': 'Lag',
  'params': ['Close', 5],
  'transform': ['Normalized', 20]},
 'f9': {'fname': 'ChaikinADOSC',
  'params': [4, 10],
  'transform': ['Normalized', 20]}}

dict_name = gen_dict_name()
list_of_input_dicts.append(dict_name)
dSet.save_pickle(test_input_dict, current_directory, dict_name)

#input_dict = dSet.load_pickle(current_directory, dict_name)
#print(input_dict)
program_name = "IS_Day_by_day_TLT_v1.py"

for dict_name in list_of_input_dicts:
    print(dict_name)

    print('python {0} {1} "{2}"'.format(program_name,'""',dict_name))
    #print(list_of_input_dicts[i])
    os.system('python {0} {1} "{2}"'.format(program_name,'""',dict_name))
    #os.system('python IS_Day_by_day_TLT_v1.py "" "input_dict_20190308_150"')
