#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:17:32 2017

@author: marc
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn2.utils import ratio2int, todf


class TruncatedSVD2(BaseEstimator, TransformerMixin):

    def __init__(self, k=2):
        self.k = k

    def _check_params(self, X):
        p = X.shape[1]
        self.k_int = ratio2int(p, self.k)
        self.donothing = (self.k_int <= 0 or self.k_int >= p)

    def fit(self, X, y=None):
        self._check_params(X)
        if self.donothing:
            return self
        self.svd = TruncatedSVD(n_components=self.k_int)
        self.svd.fit(X)
        return self

    def transform(self, X):
        return X if self.donothing else todf(self.svd.transform(X))

    def fit_transform(self, X):
        self._check_params(X)
        if self.donothing:
            return self
        self.svd = TruncatedSVD(n_components=self.k_int)
        return todf(self.svd.fit_transform(X))
