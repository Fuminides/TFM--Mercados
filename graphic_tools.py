# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:22:03 2018

@author: Javier Fumanal Idocin
"""

import pandas
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

from ggplot import ggplot, geom_line, aes, xlab,ylab, ggtitle
from matplotlib import colors as mcolors

def plot_line(X,y,title=None,labelx=None,labely=None,save=False, colors=None):
    '''
    Show on screen a line plot. Can save to a .pdf file too if specified.
    
    X,y - 
    '''
    df = pandas.DataFrame()
    
    if (title!=None):
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

def plotly_line_plot(df, columna):
    '''
    
    A simple plotly plot thay creates a line plot with a variable from a data frame
    '''

    data = [
    go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df[columna]
    )
    ]   
    return py.plot(data, filename=df['ticker'][0].replace('/', "_")+ columna)

def full_plot(df, segmentos):
    '''
    
    A simple plotly plot thay creates a line plot with a variable from a data frame
    '''
    lineas_divisorias = []
    escala = max(df.drop('volumen',axis=1).select_dtypes(include=[np.number]).max())
    for i in segmentos:
        coordenada = i[1]
       # Line Vertical
        linea = {
            'type': 'line',
            'x0': df.index[coordenada],
            'y0': 0,
            'x1': df.index[coordenada],
            'y1': escala,
            'line': {
                'color': 'rgb(255, 0, 0)',
                'width': 3,
                'dash':'dashdot'
            },
        }
        lineas_divisorias.append(linea)
        
    cierre = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['cierre']
    )
    apertura = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['apertura']
    )
    
    minimo = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['minimo']
    )
    
    maximo = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['maximo']
    )
    
    volumen = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['volumen']/np.max(df['volumen']) * escala
    )
    vtitle = 'Volumen (Factor de escala: ' + str(escala / np.max(df['volumen']))+ ')'
    data = [cierre, apertura, minimo, maximo, volumen]
    updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([   
            dict(label = 'Cierre',
                 method = 'update',
                 args = [{'visible': [True, False, False, False, False]},
                         {'title': 'Cierre'}]),
            dict(label = 'Apertura',
                 method = 'update',
                 args = [{'visible': [False, True, False, False, False]},
                         {'title': 'Apertura'}]),
            dict(label = 'Mínimo',
                 method = 'update',
                 args = [{'visible': [False, False, True, False, False]},
                         {'title': 'Minimo'}]),
            dict(label = 'Máximo',
                 method = 'update',
                 args = [{'visible': [False, False, False, True, False]},
                         {'title': 'Maximo'}]),
            dict(label = 'Volumen',
                 method = 'update',
                 args = [{'visible': [False, False, False, False, True]},
                         {'title': vtitle}])
        ]),
    )
    ])
    layout = dict(title='Financial records', showlegend=False,
              updatemenus=updatemenus,
              shapes=lineas_divisorias)
    fig = dict(data=data, layout=layout)
    
    return py.plot(fig, filename=df['ticker'][0].replace('/', "_"))

    
    