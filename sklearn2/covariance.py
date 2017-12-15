
import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.multiclass import type_of_target

from utils import get_cols


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
    return m.score(x, y) if mtype == 'linear' else m.oob_score_


def entropy(x):
    return stats.entropy(x.value_counts())


def _index_encode(y):
    classes_, encoded = np.unique(y, return_inverse=True)
    return encoded


def _compute_pearson_correls(df, columns):
    for c in get_cols(df, ['object', 'date']):
        df[c] = _index_encode(df[c])
    cor = np.corrcoef(df, rowvar=0)
    return pd.DataFrame(cor, index=df.columns, columns=df.columns)[columns]


def _compute_entropy_correls(df, columns):
    # TODO optimize
    res = pd.DataFrame(index=df.columns, columns=columns)
    for i in df.columns:
        xi = df.loc[:, i].astype(str)
        for j in columns:
            xj = df[:, j].astype(str)
            ei, ej, eij = entropy(xi), entropy(xj), entropy(xi + '-' + xj)
            mi = ei + ej - eij
            res.loc[i, j] = mi / ej
    return res.astype(float)


def _compute_model_correls(df, model, columns, sparse):
    for c in get_cols(df, ['date']):
        df[c] = _index_encode(df[c])
    res = pd.DataFrame(index=df.columns, columns=columns)
    for i in df.columns:
        xi = pd.get_dummies(df.loc[:, i], sparse=sparse)
        for j in columns:
            xj = df[:, j]
            m = _get_model(model, xj)
            m.fit(xi, xj)
            res.loc[i, j] = _get_score(model, xi, xj)
            del(m)
    return res.astype(float)


def compute_correls(df, model='pearson', columns=None, sparse=False):
    """
    Returns a 'correlation' matrix for columns of the dataframe

    model: correlation model used.

    'pearson' : standard pearson correlation, with index encoding
    for categoricals.

    'entropy' : entropy correlation ( not symetrical ).

    .. math::
        C_{ent}(X, Y) = \\frac {MI(X,Y)} {H(Y)} = 1 - \\frac {H(Y | X)} {H(Y)}

    'linear' : m.fit(x, y).score(x, y)
    m is linear or logistic regression depending on y

    'forest': random forest out of bag score
    m is ranndom forest regression or classifier depending on y


    Apart from pearson, all models produce an asymetrical matrix
    where each cell indicates how much X (line) => Y (column)

    columns : if provided, only compute correls against these columns
    """

    # TODO do not assume dataframe class
    assert isinstance(df, pd.Dataframe)
    assert model in 'pearson entropy linear forest'.split(), 'model type not implemented'

    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]

    if model == 'pearson':
        return _compute_pearson_correls(df, columns)

    if model == 'entropy':
        return _compute_entropy_correls(df, columns)

    return _compute_model_correls(df, model, columns, sparse)
