# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:23:46 2017

@author: mvanoudh
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn2.utils import ratio2int, todf

     
class TruncatedSVD2(BaseEstimator, TransformerMixin):
    """ TruncatedSVD """
    def __init__(self, k=2):
        self.k = k

    def _check_params(self, X):
        p = X.shape[1]
        self.k_int = ratio2int(p, self.k)
        self.donothing = (self.k_int <= 0 or self.k_int >= p)
        if self.donothing:
            self.feature_names = todf(X).columns
        else:
            self.feature_names = ['svd' + str(i) for i in range(self.k_int)]
                     
    def fit(self, X, y=None):
        self._check_params(X)
        if self.donothing:
            return self
        self.svd = TruncatedSVD(n_components=self.k_int)
        self.svd.fit(X)
        return self
        
    def transform(self, X):
        if self.donothing:
            return X
        return todf(self.svd.transform(X))
        
    def fit_transform(self, X, y=None):
        self._check_params(X)
        if self.donothing:
            return X
        self.svd = TruncatedSVD(n_components=self.k_int)
        return todf(self.svd.fit_transform(X))
    
    def get_feature_names(self):
        return self.feature_names
