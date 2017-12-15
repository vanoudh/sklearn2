#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:56:08 2017

@author: marc
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
        y = y.apply(lambda z: bunch['target_names'][z])
    return x, np.ravel(y)


def get_iris():
    return _extract(load_iris())


def get_boston():
    return _extract(load_boston())


def get_titanic(split=True):
    df = pd.read_csv(path.join(_here, 'titanic.csv'), sep=',',
                     encoding='latin1')
    df.columns = [clean_column(c) for c in df.columns]
    df.fare = df.fare.astype(float)
    df = df.drop('passangerid name'.split(), axis=1)
    return split_xy(df, 'survived') if split else df
