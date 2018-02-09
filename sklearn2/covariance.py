# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 15:30:54 2016
@author: mvanoudh
"""

from __future__ import unicode_literals
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.multiclass import type_of_target

from sklearn2.utils import object_cols, date_cols


def _get_model(mtype, y):
    assert mtype in ['linear', 'forest'], 'model type not implemented'
    isclass = type_of_target(y) in ['binary', 'multiclass']
    if mtype == 'linear':
        Mc = LogisticRegression if isclass else LinearRegression
        return Mc()
    else:
        Mc = RandomForestClassifier if isclass else RandomForestRegressor
        return Mc(n_estimators=40, oob_score=True, random_state=0)


def _get_score(mtype, m, x, y):
    if mtype == 'linear':
        return m.score(x, y)
    else:
        return m.oob_score_


def entropy(x):
    return stats.entropy(x.value_counts())


def _index_encode(y):
    classes_, encoded = np.unique(y, return_inverse=True)
    return encoded


def _compute_pearson_correls(df, columns):
    for c in object_cols(df) + date_cols(df):
        df[c] = _index_encode(df[c])
    cor = np.corrcoef(df, rowvar=0)
    res = pd.DataFrame(cor, index=df.columns, columns=df.columns) 
    return res[columns]

        
def _compute_entropy_correls(df, columns):
    # TODO : optimize ( lots of redundant computations )
    res = pd.DataFrame(index=df.columns, columns=columns)    
    for i in df.columns:   
        xi = df.loc[:, i].astype(str)
        for j in columns:
            xj = df.loc[:, j].astype(str)
            ei, ej, eij = entropy(xi), entropy(xj), entropy(xi + '-' + xj)
            mi = ei + ej - eij
            res.loc[i, j] = mi / ej
    return res.astype(float)
    

def _compute_model_correls(df, model='linear', columns=None, sparse=False):
    for c in date_cols(df):
        df[c] = _index_encode(df[c])
    res = pd.DataFrame(index=df.columns, columns=columns)
    for i in df.columns:   
        xi = pd.get_dummies(df.loc[:, i], sparse=sparse)
        for j in columns:
            if j == i:
                res.loc[i, j] = 1.
            else:                
                xj = df.loc[:, j]
                rf = _get_model(model, xj)
                rf.fit(xi, xj)
                res.loc[i, j] = _get_score(model, rf, xi, xj)
                del(rf)
    return res.astype(float)  # not sure why this is needed


def compute_correls(df, model='pearson', columns=None, sparse=False):
    """
    Returns a 'correlation' matrix for columns of the dataframe.
    
    model: correlation model used.
        
    'pearson' : standard pearson correl, with index encoding of
    categoricals.
    
    'entropy' : entropy correlation is used (not symetrical).
    
    .. math::
        C_{ent}(X, Y) = \\frac {MI(X,Y)} {H(Y)} = 1 - \\frac {H(Y | X)} {H(Y)}
    
    'linear' : m.fit(x, y).score(x, y) is used. 
    m is LinearRegression or LogisticRegression depending on y
           
    'forest' : m.fit(x, y).oob_score is used. 
    m is random forest.
    
    Apart from pearson, all models give an asymetrical matrix
    where each cell indicates how much X (line) => Y (column)
    
    columns: if provided, only compute correls against these columns
    """

    # TODO : do not assume on DataFrame class
    assert(isinstance(df, pd.DataFrame))
    assert model in 'pearson entropy linear forest'.split(), 'model type not implemented'
    
    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]
    
    if model=='pearson':
        return _compute_pearson_correls(df, columns)

    if model=='entropy':
        return _compute_entropy_correls(df, columns)
    
    return _compute_model_correls(df, model, columns, sparse)