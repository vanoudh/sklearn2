# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:10:21 2017

@author: mvanoudh
"""

#import numpy as np
import pandas as pd

from sklearn2.utils import print_summary

pd.options.display.width = 160


#df = pd.read_csv('titanic.csv', sep=',', encoding='latin1')
#df.columns = [clean_column(c) for c in df.columns]
#df.fare = df.fare.astype(float)
#df.age.fillna(df.age.median(), inplace=True)
#df.cabin.fillna('-', inplace=True)
#df.embarked.fillna('-', inplace=True)
#df = df.drop(['passengerid', 'name'], axis=1)

def test_run():
    df = pd.DataFrame({'a':[1, 2], 'b':['bonjour x','hello y']})
    df['c'] = df.b.apply(lambda s:s.split(' '))
    print_summary(df)
