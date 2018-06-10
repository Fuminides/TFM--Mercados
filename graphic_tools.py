# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:22:03 2018

@author: Javier Fumanal Idocin
"""

import pandas
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np


from ggplot import ggplot, geom_line, aes, xlab,ylab, ggtitle, geom_point, theme, element_text
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA
from clustering import minmax_norm

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
    
    if colors is None:
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

    df = df.iloc[0:df.shape[0]-1, :]    
    p = ggplot(df, aes(x='X'))
    
    for i in range(y.shape[1]):
         if colors not in X.columns.values:
            p = p + geom_line(aes(y=str(i),color = colors[i]))
         else:
            p = p + geom_point(aes(y=str(i),color = colors))
    
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

def full_plot(df, segmentos, clusters=[]):
    '''
    A plotly plot thay creates a line plot with a variable from a data frame
    '''
    invisibles = [True, True, True, True, True]
    lineas_divisorias = []
    
    if len(clusters) == 0:
        escala = max(df.drop('volumen',axis=1).select_dtypes(include=[np.number]).max())
    else:
        escala = max(df.drop(['volumen','cluster'],axis=1).select_dtypes(include=[np.number]).max())
    
        
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
        y=df['cierre'],
        name = "Cierre",
        mode='line'
    )
        
    apertura = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['apertura'],
        name = "Apertura",
        mode='line'
    )
    
    minimo = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['minimo'],
        name = "Mínimo",
        mode='line'
    )
    
    maximo = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['maximo'],
        name = "Máximo",
        mode='line'
        
    )
    
    volumen = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['volumen']/np.max(df['volumen']) * escala,
        name = "Volumen",
        mode='line'
    )
    
    vcierre = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['vcierre'],
        visible = "legendonly",
        name = "Ev. Cierre",
        opacity = 0
    )
    vapertura = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['vapertura'],
        visible = "legendonly",
        opacity = 0,
        name = "Ev. Apertura"
    )
    
    vminimo = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['vminimo'],
        visible = "legendonly",
        opacity = 0,
        name = "Ev. Mínimo"
    )
    
    vmaximo = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['vmaximo'],
        visible = "legendonly",
        opacity = 0,
        name = "Ev. Máximo"
    )
    
    vvolumen = go.Scatter(
        x=df['fecha'], # assign x as the dataframe column 'x'
        y=df['vvolumen'],
        visible = "legendonly",
        opacity = 0,
        name = "Ev. Volumen"
    )
    
    vtitle = 'Volumen (Factor de escala: ' + str(escala / np.max(df['volumen']))+ ')'
    data = [cierre, apertura, minimo, maximo, volumen, vcierre, vapertura, vminimo, vmaximo, vvolumen]
    updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
             dict(label = 'Todas',
             method = 'update',
             args = [{'visible': [True, True, True, True, True] + invisibles},
                     {'title': 'Todos los valores'}]),
            dict(label = 'Precio de Cierre',
                 method = 'update',
                 args = [{'visible': [True, False, False, False, False]+ invisibles},
                         {'title': 'Precio de Cierre'}]),
            dict(label = 'Precio de Apertura',
                 method = 'update',
                 args = [{'visible': [False, True, False, False, False]+ invisibles},
                         {'title': 'Precio de Apertura'}]),
            dict(label = 'Precio Mínimo',
                 method = 'update',
                 args = [{'visible': [False, False, True, False, False]+ invisibles},
                         {'title': 'Precio Mínimo'}]),
            dict(label = 'Precio Máximo',
                 method = 'update',
                 args = [{'visible': [False, False, False, True, False]+ invisibles},
                         {'title': 'Precio Máximo'}]),
            dict(label = 'Volumen',
                 method = 'update',
                 args = [{'visible': [False, False, False, False, True]+ invisibles},
                         {'title': vtitle}])
        ]),
    )
    ])
    nombre_grafo = 'Registro Financiero de ' + df['ticker'][0][df['ticker'][0].find("/")+1:]
    layout = dict(title=nombre_grafo, showlegend=False,
              updatemenus=updatemenus,
              shapes=lineas_divisorias)
    fig = dict(data=data, layout=layout)
    
    return py.plot(fig, filename=df['ticker'][0].replace('/', "_"))

def visualize_clusters(X, var):
    '''
    Prints with ggplot a visualization of the different clusters.
    '''
    aux = pandas.DataFrame()
    
    aux['fecha'] = X.index
    aux.index = X.index
    
    aux[var] = X[var]
    aux['Cluster'] = X['cluster']
    
    return ggplot(aes(x='fecha', y=var, color='Cluster'), aux) + geom_point() + xlab(var) + ylab("Valor") + ggtitle("Clustering de la variable \"" + var + "\"") +  theme(axis_text_x  = element_text(color=[0,0,0,0]))

def visualize_segmentation(X, var):
    '''
    Prints with ggplot a visualization of the different segments.
    '''
    aux = pandas.DataFrame(index = X.index)
    
    aux['fecha'] = X.index.values
    aux[var] = X[var]
    aux['Segmento'] = X['segmento'].astype(str)
    
    return ggplot(aes(x="fecha", y=var, color="Segmento"), aux) + geom_line() + xlab("Fecha") + ylab(var) + ggtitle("Segmentacion de la variable \"" + var + "\"")
    

def biplot(X):
    '''
    Prints a biplot with ggplot. Requires color variable: "cluster" in the dataframe.
    '''
    pca = PCA(n_components=2)
    try:
        res = pca.fit_transform(minmax_norm(X).drop(['fecha','ticker','cluster'], axis=1))
    except ValueError:
        res = pca.fit_transform(minmax_norm(X))
        
    df = pandas.DataFrame(res)
    df.columns = ["x", "y"]
    df['cluster'] = X['cluster'].values
    
    return ggplot(aes("x","y", color="cluster"),df) + geom_point()

def plotly_biplot(X):
    '''
    Prints a biplot with plotly. Requires color variable: "cluster" in the dataframe.
    '''
    pca = PCA(n_components=2)
    try:
        res = pca.fit_transform(minmax_norm(X).drop(['fecha','ticker'], axis=1))
    except ValueError:
        res = pca.fit_transform(minmax_norm(X))
    df = pandas.DataFrame(res)
    df.columns = ["x", "y"]
    df['cluster'] = X['cluster'].values
    
    cierre = go.Scatter(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y'],
        mode='markers',
        marker=dict(
            color = df['cluster'],
            colorscale = 'Pastel2',
        )
    )
        
    data = [cierre]
    layout = dict(title="Biplot", showlegend=False)
    fig = dict(data=data, layout=layout)
        
    return py.plot(fig, filename="Biplot")
    
    
    
    