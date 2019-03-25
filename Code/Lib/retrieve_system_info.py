# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:06:54 2019

@author: KRUEGKJ
"""

import random
import os
import os.path
import sys
from Code.lib.retrieve_data import DataRetrieve
dSet = DataRetrieve()

class TradingSystemUtility:
    # Function to generate system number
    def get_sys_number(self):
        # Generate four digit random number
        a = random.sample(range(10),3)
        number = int(str(random.randint(1,9)) + str(a[0]) + str(a[1]) + str(a[2]))
        return number
    
    # function to create system directory based on system name if one doesn't already exist
    def get_system_dir(self, system_name):
        # Create system directory in current path
        current_directory = os.getcwd()
        system_directory = os.path.join(current_directory, system_name)
        if not os.path.exists(system_directory):
           os.makedirs(system_directory)
        return system_directory
    
    # function to get system name
    def get_system_name(self, issue, direction, sys_no, ver_num):
        system_name = issue + "-" + direction + "-system-" + str(sys_no) + "-V" + str(ver_num)
        return system_name
    
    # function to get system_dict
    def get_system_dict(self, system_name, issue, direction, ver_num):
        if system_name == "":
            sys_no = self.get_sys_number()
            system_name = self.get_system_name(issue, direction, sys_no, ver_num)
            print(system_name)
            system_directory = self.get_system_dir(system_name)
            # Create system_dict
            system_dict = {}
            system_dict = {'issue': issue,
                           'direction': direction,
                           'system_name': system_name,
                           'ver_num': ver_num
                           }
            self.save_dict(system_directory, 'system_dict', system_dict)
            return system_dict
        else:
            system_directory = self.get_system_dir(system_name)
            system_dict = self.get_dict(system_directory, 'system_dict')
            return system_dict
    
    def get_dict(self, system_directory, dict_name):
        dict_lookup = {'system_dict': 'system_dict.json',
                       'feature_dict': 'feature_dict.json',
                       'input_dict': 'input_dict.pkl',
                       'tms_dict': 'feature_dict.json'}
        
        file_name = dict_lookup[dict_name]
        fn_split = file_name.split(".",1)
        file_suffix = fn_split[1]
        if file_suffix == 'json':
            return_dict = dSet.load_json(system_directory, file_name)
        elif file_suffix == 'pkl':
            return_dict = dSet.load_pickle(system_directory, file_name)
        else:
            print('dict type not found')
            sys.exit()
        return return_dict
    
    def save_dict(self, system_directory, dict_name, dict_file):
        dict_lookup = {'system_dict': 'system_dict.json',
                       'feature_dict': 'feature_dict.json',
                       'input_dict': 'input_dict.pkl',
                       'tms_dict': 'feature_dict.json'}
        
        file_name = dict_lookup[dict_name]
        fn_split = file_name.split(".",1)
        file_suffix = fn_split[1]
        if file_suffix == 'json':
            dSet.save_json(file_name, system_directory, dict_file)
            print(file_name + ' saved.')
        elif file_suffix == 'pkl':
            dSet.save_pickle(file_name, system_directory, dict_file)
            print(file_name + ' saved.')
        else:
            print('dict type not found')
            sys.exit()
            
if __name__ == "__main__":
    sysUtil = TradingSystemUtility()
    
    # test for new system
    issue = "TLT"
    direction = "Long"
    ver_num = 1
    
    sys_no = sysUtil.get_sys_number()
    print(sys_no)
    system_name = sysUtil.get_system_name(issue, direction, sys_no, ver_num)
    print(system_name)
    system_dir = sysUtil.get_system_dir(system_name)
    print(system_dir)
    
    # new
    system_dict = sysUtil.get_system_dict("", issue, direction, ver_num)
    print(system_dict)
    
    # existing
    system_name = system_dict['system_name']
    exist_system_dict = sysUtil.get_system_dict(system_name, issue, direction, ver_num)
    print(exist_system_dict)
    
    
    
            