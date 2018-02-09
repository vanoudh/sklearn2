# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 15:30:54 2016
@author: mvanoudh
"""

#import logging
import pandas as pd
import numpy as np
import unicodedata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import issparse

pd.options.display.width = 160

               
def todf(X, cols=None):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=cols)


def numeric_cols(df):
    types = ['float64', 'int64', 'float32', 'int32']
    return [c for c in df.columns if df[c].dtype in types]


def object_cols(df):
    types = ['object']
    return [c for c in df.columns if df[c].dtype in types]


def bool_cols(df):
    types = ['bool']
    return [c for c in df.columns if df[c].dtype in types]


def date_cols(df):
    types = ['datetime64[ns]']
    return [c for c in df.columns if df[c].dtype in types]


def not_date_cols(df):
    types = ['datetime64[ns]']
    return [c for c in df.columns if df[c].dtype not in types]


def print_summary(df, width1=40, width2=80):
    """ print a nice summary of a pd.DataFrame """

    df = todf(df)

    lc = max([len(str(c)) for c in df.columns])
    width1 = min(width1, lc)
    
    HEADER = ' '*(width1-6) + 'column |    nulls |   unique | type           | mode/med'
    FORMAT = '%' + str(width1) + 's | %6d   | %6d   | %14s | %s'
#    FORMAT = '%s | %6d   | %6d   | %14s | %s'
    #HEADER = '                                              column |    nulls |   unique | type           | most common'
    #FORMAT = '%52s | %6d   | %6d   | %14s | %s'
    print('%d lines - %d columns' % (len(df), df.shape[1]))
    print(HEADER)
    print('-'*width2)
    for col in df.columns:
        nuni = -999
        mode = -999
        try:
            nuni = df[col].nunique()
            if df[col].dtype in ['float32', 'float64']:
                mode = df[col].median()
            else:
                z = df[col].value_counts()[:1]
                mode = '{} ({})'.format(z.index[0], z.iloc[0])
        except:
            pass
        s = FORMAT % (str(col)[:width1], df[col].isnull().sum(),
                      nuni, df[col].dtype, mode)
        print(s[:width2])


def print_summary2(df):
    pres = pd.DataFrame()
    pres['nulls'] = df.isnull().sum()
    pres['unique'] = df.nunique()
    pres['type'] = df.dtypes
    pres['mode'] = df.mode(axis=0).iloc[0]
    pres['median'] = df.median()
    print(pres)

    
def model_name(m):
    return str(m).split('(')[0].split('.')[-1].split(' ')[0]

 
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

   
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

    
def clean_text(s):
    if s is None:
        return ''
    r = strip_accents(s.strip().lower())
    for c in "\n\t'":
        r = r.replace(c, ' ')
    return r

                  
def align_columns(df, target_columns):
    if list(df.columns) == list(target_columns):  # nothing to do
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
    node_indicator = estimator.decision_path(X)
    leaf_nodes = estimator.apply(X)
    threshold = estimator.tree_.threshold    
    value = estimator.tree_.value
    
    for sample_id in range(len(X)):
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        
        print('\nRules used to predict sample %s: ' % sample_id)
        for node_id in node_index:
            probas = value[node_id][0] / np.sum(value[node_id][0])
            ic = np.argmax(probas)
            print('{} at {:.0%}'.format(estimator.classes_[ic], probas[ic]))
            if (leaf_nodes[sample_id] == node_id):
                continue
            feature_i = estimator.tree_.feature[node_id]
            thresh = threshold[node_id]
            sign = "<=" if (X[sample_id, feature_i] <= thresh) else ">"
            print("decision id node %d : %s = %s %s %s"
                  % (node_id, feature_names[feature_i],
                     X[sample_id, feature_i], sign, thresh))

            


def get_forest(y):
    isclass = type_of_target(y) in ['binary', 'multiclass']
    return RandomForestClassifier if isclass else RandomForestRegressor


def feature_importance(x, y, n=20):
    rf = get_forest(y)(n_estimators=100)
    rf.fit(x, y)
    res = pd.Series(rf.feature_importances_, index=x.columns)
    return res.sort_values(ascending=False).head(n)

    
def forest_sort(x, y):
    x_df = todf(x)
    rf = get_forest(y)(n_estimators=100)
    rf.fit(x, y)
    imp = pd.Series(data=rf.feature_importances_, index=x_df.columns)
    imp.sort_values(ascending=False, inplace=True)
    return x_df[imp.index]   


def ratio2int(p, k):
    """Returns p * k if k < 1 else k
    
    Result is cast to nearest integer in [1, p]
        
    Parameters:
    -----------
    p : int
        
    k : int or float
        
    >>> ratio2int(10, 1)
    1
    >>> ratio2int(10, 1.0)
    1
    >>> ratio2int(10, .99)
    10
    >>> ratio2int(12, 3)
    3
    >>> ratio2int(10, 15)
    10
    >>> ratio2int(10, 0)
    1
    >>> ratio2int(10, 0.5)
    5
    >>> ratio2int(10, 0.1)
    1
    >>> ratio2int(10, 0.01)
    1
    >>> ratio2int(87, np.float64(0.8))
    70
    """
    return min(max(int(round(p * k if k < 1 else k)), 1), p)

    
class EstimatorWrap:
    """ Wrapper for estimators that do not accept dataframes """
    
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, x, y):
        r = self.estimator.fit(x.as_matrix(), y.as_matrix())
        self.classes_ = self.estimator.classes_
        return r
        
    def predict(self, x):
        return self.estimator.predict(x.as_matrix())
        
    def predict_proba(self, x):
        return self.estimator.predict_proba(x.as_matrix())
        
    def score(self, x, y):
        return self.estimator.score(x.as_matrix(), y.as_matrix())

        
class TransformerWrap(BaseEstimator, TransformerMixin):
    """ Wrapper for transformers that 'downcast' dataframes """
    
    def __init__(self, transformer):
        self.transformer = transformer
                
    def fit(self, X, y=None, **fit_params):
        self.cols = X.columns if isinstance(X, pd.DataFrame) else None
        self.transformer.fit(X, y)
        return self
        
    def transform(self, X, y=None, **fit_params):
        return todf(self.transformer.transform(X), self.cols)

    def fit_transform(self, X, y=None, **fit_params):
        self.cols = X.columns if isinstance(X, pd.DataFrame) else None
        return todf(self.transformer.fit_transform(X), self.cols)

        
class SelectorWrap(BaseEstimator, TransformerMixin):
    
    def __init__(self, selector):
        self.selector = selector
                
    def fit(self, X, y=None, **fit_params):
        self.selector.fit(X, y)
        self.support = self.selector.get_support()
        return self
        
    def transform(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support] # ok for DataFrame
        else:            
            return X[:, self.support] # ok for np.array and scipy sparse

   
class DenseTransformer(BaseEstimator, TransformerMixin):
    """ Make a matrix dense """
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
        
    def transform(self, X, y=None, **fit_params):
        return X.toarray() if issparse(X) else X

            
class PassThrought(BaseEstimator, TransformerMixin):
    """ dummy transformer that doesn't do anything """
    def __init__(self):
        self.feature_names = None
    
    def fit(self, X, y=None, **fit_params):
        return self
        
    def transform(self, X, y=None, **fit_params):
        self.feature_names = X.columns
        return X
        
    def fit_transform(self, X, y=None, **fit_params):
        self.feature_names = X.columns
        return X

    def get_feature_names(self):
        return self.feature_names
   
     
#Just wanted to toss out the solution I am using:
#
#def check_output(X, ensure_index=None, ensure_columns=None):
#    """
#    Joins X with ensure_index's index or ensure_columns's columns when avaialble
#    """
#    if ensure_index is not None:
#        if ensure_columns is not None:
#            if type(ensure_index) is pd.DataFrame and type(ensure_columns) is pd.DataFrame:
#                X = pd.DataFrame(X, index=ensure_index.index, columns=ensure_columns.columns)
#        else:
#            if type(ensure_index) is pd.DataFrame:
#                X = pd.DataFrame(X, index=ensure_index.index)
#    return X
#I then create wrappers around sklearn's estimators that call this function on the output of transform e.g.,
#
#from sklearn.preprocessing import StandardScaler as _StandardScaler 
#class StandardScaler(_StandardScaler):
#    def transform(self, X):
#        Xt = super(StandardScaler, self).transform(X)
#        return check_output(Xt, ensure_index=X, ensure_columns=X)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
