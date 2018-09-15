# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 22:30:55 2018

@author: KRUEGKJ
"""

def print1():
    print(1)
    
def print2():
    print(2)

ops = {
"1" : print1,
"2" : print2

}

for key in ["1","2"]:
    ops[key]()