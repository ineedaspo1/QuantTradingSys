# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:04:07 2019

@author: kruegkj

Re-use existing input_dicts to generate new results
"""

import os
import glob

from Code.lib.retrieve_data import DataRetrieve

current_directory = os.getcwd()

dSet = DataRetrieve()



# create  
list_of_input_dicts = []

num_dicts_to_make = 40
dict_ctr = 0

dicts_to_use = glob.glob("input_dict*")

program_name = "IS_Day_by_day_TLT_v1.py"

for dict_name in dicts_to_use:
    print(dict_name)
    current_directory = os.getcwd()
        # add try/catch block later
    filename = dict_name
    input_dict = dSet.load_pickle(current_directory, filename)
    print(input_dict)
    
    print('python {0} {1} "{2}"'.format(program_name,'""',dict_name))
    #print(list_of_input_dicts[i])
    os.system('python {0} {1} "{2}"'.format(program_name,'""',dict_name))
