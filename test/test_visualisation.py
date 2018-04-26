# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:10:21 2017

@author: mvanoudh
"""

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  #analysis:ignore
import numpy as np
import pandas as pd
from numpy.random import seed

from sklearn2.visualisation import show_heatmap

    
seed(0)
pd.options.display.width = 160


def test_heatmap():
    print(os.getcwd())
    res1 = pd.DataFrame(np.random.rand(5, 5))
    fig = plt.figure(figsize=(8,6))      
    show_heatmap(res1)
    plt.xlabel('line => column')
    os.makedirs('test_out', exist_ok=True)
    fig.savefig(os.path.join('test_out', 'heatmap.png'))

    
if __name__ == "__main__":
    test_heatmap()