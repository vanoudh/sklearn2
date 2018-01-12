# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 15:30:54 2016
@author: mvanoudh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


from sklearn2.utils import numeric_cols, object_cols

               
def show_histo(df, bins=20):
    """ plot histograms of columns """

    assert(isinstance(df, pd.DataFrame))

    for c in numeric_cols(df):
        df[c].hist(bins=bins)
        plt.title(c)
        plt.show()
        

def show_freq(df, max_card=20):
    """ plot histograms of columns """

    assert(isinstance(df, pd.DataFrame))

    for c in object_cols(df):
        if df[c].nunique() < max_card:
            df[c].value_counts().plot(kind='barh')
            plt.title(c)
            plt.show()


def show_heatmap(data, columns=None, index=None, ax=None):
    """ plot the heatmap of the data

    Parameters:
    -----------
    data : np.array of pd.Dataframe
        the numbers to put in the heatmap

    columns: list of str or None
        the names of the columns, if None will use the columns of the DataFrame

    index : list of str or None
        the name of the lines, if None will use the index of the DataFrame
    ax : where to plot the heatmap
    """

    if ax is None:
        # fig, ax = plt.subplots() # nouvelle figure
        ax = plt.gca()  # axe courrant

    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = list(data.columns)
        if index is None:
            index = list(data.index)
        data = data.values

    ax.cla()
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn, edgecolors='k')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i+0.5, "%2.2f" % data[i, j])

    # Affiche la colorbar a cote
    # Attention ca rajoute un ax a cote de l'axe courant
    plt.colorbar(heatmap, ax=ax)

    # change les ticks
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    ax.invert_yaxis()  # retourne axe des y
    ax.xaxis.tick_top()

    ax.set_xticklabels(columns, minor=False, rotation=90)
    ax.set_yticklabels(columns, minor=False)
    ax.set_xlim((0, data.shape[0]))
    ax.set_ylim((data.shape[1], 0))
    return heatmap


def show_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
from sklearn.externals.six import StringIO  
from pydot import graph_from_dot_data
from sklearn.tree import export_graphviz


def dump_tree_class(clf, feature_names, filename):
    dot_data = StringIO() 
    export_graphviz(clf, out_file=dot_data, feature_names=feature_names, 
                    class_names=clf.classes_)
    graph = graph_from_dot_data(dot_data.getvalue())[0]
    graph.write_pdf(filename + '.pdf')

    
def dump_tree_reg(reg, feature_names, filename):
    dot_data = StringIO() 
    export_graphviz(reg, out_file=dot_data, feature_names=feature_names)
    graph = graph_from_dot_data(dot_data.getvalue())[0]
    graph.write_pdf(filename + '.pdf')