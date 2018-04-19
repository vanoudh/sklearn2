# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:46:47 2017

@author: mvanoudh
"""

from os import path
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn2.utils import split_xy

_here = path.abspath(path.dirname(__file__))

def _extract(bunch, return_xy):
    x = pd.DataFrame(data=bunch['data'], columns=bunch['feature_names'])
    y = pd.DataFrame(data=bunch['target'], columns=['target'])
    if 'target_names' in bunch.keys():
        y = y.apply(lambda z:bunch['target_names'][z])
    return x, np.ravel(y) if return_xy else pd.concat([x, y], axis=1)


def get_iris(return_xy=False):
    return _extract(load_iris(), return_xy)


def get_boston(return_xy=False):
    return _extract(load_boston(), return_xy)


def get_titanic(return_xy=False):
    df = pd.read_csv(path.join(_here, 'titanic.csv'), encoding='latin1')
    df.Fare = df.Fare.astype(float)
    return split_xy(df, 'Survived') if return_xy else df


def get_titanic_clean(return_xy=False):
    df = pd.read_csv(path.join(_here, 'titanic.csv'), encoding='latin1')
    df.Fare = df.Fare.astype(float)
    df.Age.fillna(df.Age.median(), inplace=True)
    df.Cabin.fillna('-', inplace=True)
    df.Embarked.fillna('-', inplace=True)
    return split_xy(df, 'Survived') if return_xy else df


def get_house_prices(return_xy=False):
    df = pd.read_csv(path.join(_here, 'house_prices.zip'), encoding='latin1')
    return split_xy(df, 'SalePrice') if return_xy else df
