# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:46:47 2017

@author: mvanoudh
"""

from os import path
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn2.utils import clean_column, split_xy

_here = path.abspath(path.dirname(__file__))

def _extract(bunch):
    x = pd.DataFrame(data=bunch['data'], columns=bunch['feature_names'])
    y = pd.DataFrame(data=bunch['target'], columns=['target'])
    if 'target_names' in bunch.keys():
        y = y.apply(lambda z:bunch['target_names'][z])
#    df = pd.concat([x, y], axis=1)
    return x, np.ravel(y)


def get_iris():
    return _extract(load_iris())


def get_boston():
    return _extract(load_boston())


def get_titanic(split=True, clean=True):  
    df = pd.read_csv(path.join(_here, 'titanic.csv'), encoding='latin1')
    if clean:
        df.columns = [clean_column(c) for c in df.columns]
        df.fare = df.fare.astype(float)
        df.age.fillna(df.age.median(), inplace=True)
        df.cabin.fillna('-', inplace=True)
        df.embarked.fillna('-', inplace=True)
        df = df.drop(['passengerid', 'name'], axis=1)
    return split_xy(df, 'survived') if split else df 


def get_house_prices(split=True):  
    df = pd.read_csv(path.join(_here, 'house_prices.zip'), encoding='latin1')
#    df.columns = [clean_column(c) for c in df.columns]
    df = df.drop(['Id'], axis=1).dropna(1)
    return split_xy(df, 'SalePrice') if split else df 
