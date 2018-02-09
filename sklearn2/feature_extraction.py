# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:09:57 2016

@author: mvanoudh
"""

import logging
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

from sklearn2.utils import object_cols, numeric_cols
from sklearn2.utils import date_cols, align_columns, todf

logger = logging.getLogger(__name__)

    
class DateEncoder(TransformerMixin, BaseEstimator):
    
    def __init__(self, ascategory=True, t0=pd.datetime(2000, 1, 1)):
        self.ascategory = ascategory
        self.t0 = t0

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        enc = lambda i:str(i) if self.ascategory else i
        r = X
        for c in date_cols(r):
            logger.debug('encoding date column {}'.format(c))
            r[c + '_year'] = r[c].apply(lambda ts:enc(ts.year))
            r[c + '_month'] = r[c].apply(lambda ts:enc(ts.month))
            r[c + '_week'] = r[c].apply(lambda ts:enc(ts.week))
            r[c + '_wom'] = r[c].apply(lambda ts:enc((ts.day-1) // 7 + 1))
            r[c + '_day'] = r[c].apply(lambda ts:enc(ts.day))
            r[c + '_dow'] = r[c].apply(lambda ts:enc(ts.dayofweek))
            r[c + '_hour'] = r[c].apply(lambda ts:enc(ts.hour))
            r[c] = r[c].apply(lambda ts:float(ts.toordinal() - self.t0.toordinal()))
        return r

        
class SparseCatEncoder(TransformerMixin, BaseEstimator):
    
    def __init__(self, drop_first=False, sparse=False):
        self.drop_first = drop_first
        self.sparse = sparse
        self.encoded = None
            
    def fit(self, X, y=None):
        if X.shape[1] == 0:
            self.encoded = X
            return self
        self.encoded = pd.get_dummies(todf(X), dummy_na=True,
                                      drop_first=self.drop_first, 
                                      sparse=self.sparse)            
        logger.debug('dummy encoded shape {}'.format(self.encoded.shape))
        return self
    
    def transform(self, X, y=None):
        if X.shape[1] == 0:
            return X
        encoded = pd.get_dummies(todf(X), dummy_na=True,
                                 drop_first=False, 
                                 sparse=self.sparse)
        logger.debug('dummy encoded shape {}'.format(encoded.shape))
        return align_columns(encoded, self.encoded.columns)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.encoded


class IndexCatEncoder(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.encoded = None
        self.encoders = {}
            
    def fit(self, X, y=None):
        for c in object_cols(X):
            le = LabelEncoder()
            le.fit(X[c])
            self.encoders[c] = le
        logger.debug('index encoders {}'.format(self.encoders))
        return self
    
    def transform(self, X, y=None):
        Xt = X.copy()
        for c in object_cols(X):
            Xt[c] = self.encoders[c].transform(Xt[c])
        logger.debug('')
        return Xt

    
class ConstantInputer(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        pass
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.fillna(-999)


class NumericFilter(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        pass
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[numeric_cols(X)]

