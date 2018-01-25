# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:22:03 2018

@author: Javier Fumanal Idocin
"""

import pandas

from ggplot import ggplot, geom_line, aes, xlab,ylab, ggtitle
from matplotlib import colors as mcolors

def plot_line(X,y,title=None,labelx=None,labely=None,save=False, colors=None):
    '''
    Show on screen a line plot. Can save to a .pdf file too if specified.
    
    X,y - 
    '''
    df = pandas.DataFrame()
    img_title = title.replace(" ","").replace(".","-") + ".pdf"
    
    df['X'] = X 
    for i in range(y.shape[1]):
        df[str(i)] = y.iloc[:,i].values
    
    if colors==None:
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

    df = df.iloc[0:df.shape[0]-1, :]    
    p = ggplot(df, aes(x='X'))
    
    for i in range(y.shape[1]):
         p = p + geom_line(aes(y=str(i),color = colors[i]))
    
    p = p + xlab(labelx) + ylab(labely) + ggtitle(title)
    
    if(save):
        p.save(img_title)
    else:   
        return p

    