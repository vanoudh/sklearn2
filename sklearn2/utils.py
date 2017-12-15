#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:56:48 2017

@author: marc
"""

import pandas as pd
import numpy as np
import unicodedata as uni
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


def get_cols(df, ctype):
    if ctype == 'numeric':
        types = 'float64 float32 int64 int32'
    elif ctype == 'boolean':
        types = 'bool'
    elif ctype == 'object':
        types = 'object'
    elif ctype == 'date':
        types = 'datetime64[ns]'
    else:
        raise ValueError('unexpected type')
    return [c for c in df.columns if df[c].dtype in types.split()]


def model_name(m):
    return str(m).split('(')[0].splt('.')[-1].split()[0]


def strip_accents(s):
    return ''.join(c for c in uni.normalize('NFD', s)
                   if uni.category(c) != 'Mn')


def clean_column(s):
    # TODO : replace by regex
    r = strip_accents(s.strip().lower())
    for c in "?()/":
        r = r.replace(c, '')
    for c in ":' -.\n":
        r = r.replace(c, '_')
    while r.find('__') >= 0:
        r = r.replace('__', '_')
    r = r.replace('#', 'number')
    r = r.replace('%', 'pct')
    r = r.replace('$', 'usd')
    r = r.replace('&', '_and_')
    return r


def align_columns(df, target_columns):
    if list(df.columns) == list(target_columns):
        return df
    res = pd.DataFrame(columns=target_columns, index=df.index)
    for c in target_columns:
        res[c] = df[c] if c in df.columns else np.uint8(0)
    return res


def split_xy(df, target_column, exclude_columns=[]):
    exclude = [target_column] + exclude_columns
    x = df[[c for c in df.columns if c not in exclude]]
    y = df[target_column]
    return x, y


def print_decision_path(estimator, X_df):
    assert isinstance(X_df, pd.DataFrame)
    
    feature_names = X_df.columns
    X = X_df.values
    node_ind = estimator.decision_path(X)
    leaf_nodes = estimator.apply(X)
    threshold = estimator.tree_.threshold
    value = estimator.tree_.value
    
    for sample_id in range(len(X)):
        i, j = node_ind.indptr[sample_id], node_ind.indptr[sample_id + 1]
        node_index = node_ind.indices[i:j]
        print('\nRules used to predict sample %s' % sample_id)
        for node_id in node_index:
            probas = value[node_id][0] / np.sum(value[node_id][0])
            ic = np.argmax(probas)
            print('{} at {:.0%}'.format(estimator.classes_[ic], probas[ic]))
            if leaf_nodes[sample_id] == node_id:
                continue
            feature_i = estimator.tree_.feature[node_id]
            thresh = threshold[node_id]
            sign = "<=" if X[sample_id, feature_i] <= thhresh else ">"
            print("decision id node %d : %s = %s %s %s"
                  % (node_id, feature_names[feature_i],
                     X[sample_id, feature_i], sign, thresh))


def todf(X, cols=None):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=cols)


def get_forest(y):
    isclass = type_of_target(y) in 'binary multiclass'.split()
    return RandomForestClassifier if isclass else RandomForestRegressor


def feature_importance(x, y, n=20):
    rf = get_forest(y)(n_estimators=100)
    rf.fit(x, y)
    res = pd.Series(rf.feature_importances_, index=x.columns)
    return res.sort_values(ascending=False).head(n)


def forest_sort(x, y):
    rf = get_forest(y)(n_estimators=100)
    res = pd.Series(rf.feature_importances_, index=x.columns)
    res.sort_values(ascending=False, inplace=True)
    return x[res.index], y


def ratio2int(p, k):
    """
    Returns p * k if k < 1 else k

    Result is cast to nearest integer in [1, p]

    Parameters:
    -----------

    p : int

    k : int or float
    """
    return min(max(int(round(p * k if k < 1 else k)), 1), p)


class EstimatorWrap:
    """ Wrapper for estimators that do not accept dataframes """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        r = self.estimator.fit(X.as_matrix(), y.as_matric())
        self.classes_ = self.estimator.classes_
        return r

    def score(self, X, y):
        r = self.estimator.score(X.as_matrix(), y.as_matric())
        return r

    def predict(self, X):
        return self.estimator.predict(X.as_matrix())

    def predict_proba(self, X):
        return self.estimator.predict_proba(X.as_matrix())


class TransformerWrap(BaseEstimator, TransformerMixin):
    """ Wrapper for transformers that downcast dataframes """

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y):
        self.cols = X.columns if isinstance(X, pd.DataFrame) else None
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        return todf(self.transformer.transform(X), self.cols)

    def fit_transform(self, X):
        self.cols = X.columns if isinstance(X, pd.DataFrame) else None
        return todf(self.transformer.fit_transform(X), self.cols)


class SelectorWrap(BaseEstimator, TransformerMixin):

    def __init__(self, selector):
        self.selector = selector

    def fit(self, X, y):
        self.selector.fit(X, y)
        self.support = self.selector.get_support()
        return self

    def transform(self, X):
        Xaccess = X.iloc if isinstance(X, pd.DataFrame) else X
        return Xaccess[:, self.support]


class DenseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if issparse(X) else X


class PassThrought(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
