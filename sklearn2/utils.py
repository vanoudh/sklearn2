#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:56:48 2017

@author: marc
"""

import pandas as pd
import numpy as np
import unicodedata
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import issparse

pd.options.display.width = 160

FORMAH = '%30s | %6s | %6s | %14s | %s'
FORMAT = '%30s | %6d | %6d | %14s | %s'


def print_summary(df):
    """ print a nice summary of the dataframe """

    assert isinstance(df, pd.DataFrame)

    print('%d lines - %d columns' % df.shape)
    print(FORMAH % ('column', 'nulls', 'unique', 'type', 'median/mode'))
    print('-' * 80)
    for col in df.columns:
        nuni, mode = -999, -999
        try:
            nuni = df[col].nunique()
            mode = df[col].mode()
        except:
            pass
        ctype = df[col].dtype
        print(FORMAT % (
                col,
                df[col].isnull().sum(),
                nuni,
                ctype,
                mode
                ))


