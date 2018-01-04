# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:42:49 2017

@author: mvanoudh
"""


import os
import sys


def file_filter(filename):
    if filename.endswith('.py'):
        return True
    if filename.endswith('.cfg'):
        return True
    

path = sys.argv[1]

for dirname, dirnames, filenames in os.walk(path):
    for filename in filenames:
        if file_filter(filename):
            fpath = os.path.join(dirname, filename)
            f = open(fpath, 'r')
            lines = f.readlines()
            f.close()
            print('#file:{}:{}'.format(dirname, filename))
            for line in lines:
                print(line, end='')
            print('')


