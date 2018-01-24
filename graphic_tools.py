# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:22:03 2018

@author: javi-
"""

import pandas

from ggplot import ggplot, geom_line, aes
from matplotlib import colors as mcolors

def plot_line(X,y):
    '''
    Show on screen a line plot.
    
    X,y - 
    '''
    df = pandas.DataFrame()
    
    df['X'] = X 
    for i in range(y.shape[1]):
        df[str(i)] = y.iloc[:,i].values
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

    df = df.iloc[0:df.shape[0]-1, :]    
    p = ggplot(df, aes(x='X'))
    
    for i in range(y.shape[1]):
         p = p + geom_line(aes(y=str(i),color = colors[i]))
         
    return p

    