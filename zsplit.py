# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:42:49 2017

@author: mvanoudh
"""


import os
import sys


inpath, outpath = sys.argv[1], sys.argv[2]

f = open(inpath, 'r')
fn = None
line = f.readline()
while line != '':
    ls = line.rstrip('\n').split(':')
    if ls[0] == '#file':
        dirname, filename = ls[1], ls[2]
        os.makedirs(os.path.join(outpath, dirname), exist_ok=True)
        if fn != None:
            fn.close()
        fn = open(os.path.join(outpath, dirname, filename), "w")
    else:
        fn.write(line)
    line = f.readline()
fn.close()
f.close()


