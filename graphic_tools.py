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
from clustering import minmax_norm, filter_numerical

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
    df=df.drop(['var','vvar'],axis=1,errors='ignore')
    lineas_divisorias = []
    
    if len(clusters) == 0:
        escala = max(df.drop(['volumen','vcierre','vapertura','vmaximo','vminimo','vvolumen'],axis=1,errors='ignore').select_dtypes(include=[np.number]).max())
    else:
        escala = max(df.drop(['volumen','vcierre','cluster','vapertura','vmaximo','vminimo','vvolumen'],axis=1,errors='ignore').select_dtypes(include=[np.number]).max())
    
        
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
        
    data = []
    invisibles = [True] * int(len(list(df))/2)
    botones = []
    
    tobutton = dict(label = 'Todos',
                 method = 'update',
                 args = [{'visible': [True] * int(len(list(df))/2)+ invisibles},
                         {'title': 'Todos los campos'}])
    botones.append(tobutton)
    try:
        cierre = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['cierre'],
            name = "Cierre",
            mode='line'
        )
        data.append(cierre)
        
        visuals = [False] * int(len(list(df))/2)
        visuals[len(data)-1] = True
        cbutton = dict(label = 'Precio de Cierre',
                 method = 'update',
                 args = [{'visible': visuals+ invisibles},
                         {'title': 'Precio de Cierre'}])
        botones.append(cbutton)
    except KeyError:
        pass
    try:    
        apertura = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['apertura'],
            name = "Apertura",
            mode='line'
        )
        
        data.append(apertura)
        
        visuals = [False] * int(len(list(df))/2)
        visuals[len(data)-1] = True
        abutton = dict(label = 'Precio de Apertura',
                 method = 'update',
                 args = [{'visible': visuals+ invisibles},
                         {'title': 'Precio de Apertura'}])
        botones.append(abutton)
    except KeyError:
        pass
    
    try:
        minimo = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['minimo'],
            name = "Mínimo",
            mode='line'
        )
        data.append(minimo)
        
        visuals = [False] * int(len(list(df))/2)
        visuals[len(data)-1] = True
        mibutton = dict(label = 'Precio Mínimo',
                 method = 'update',
                 args = [{'visible': visuals+ invisibles},
                         {'title': 'Precio Mínimo'}])
        botones.append(mibutton)
        
    except KeyError:
        pass    
    try:
        maximo = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['maximo'],
            name = "Máximo",
            mode='line'     
        )
        data.append(maximo)
        
        visuals = [False] * int(len(list(df))/2)
        visuals[len(data)-1] = True
        mabutton= dict(label = 'Precio Máximo',
                 method = 'update',
                 args = [{'visible': visuals+ invisibles},
                         {'title': 'Precio Máximo'}])
        botones.append(mabutton)
    except KeyError:
        pass
    
    try:
        volumen = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['volumen']/np.max(df['volumen']) * escala,
            name = "Volumen",
            mode='line'
        )
        data.append(volumen)
        
        visuals = [False] * int(len(list(df))/2)
        visuals[len(data)-1] = True
        vtitle = 'Volumen (Factor de escala: ' + str(escala / np.max(df['volumen']))+ ')'
        vbutton = dict(label = 'Volumen',
                 method = 'update',
                 args = [{'visible': visuals + invisibles},
                         {'title': vtitle}])
        botones.append(vbutton)
        
    except KeyError:
        pass
    
    try:
        vcierre = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['vcierre'],
            visible = "legendonly",
            name = "Ev. Cierre",
            opacity = 0
        )
        data.append(vcierre)
    except KeyError:
        pass
    
    try:
        vapertura = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['vapertura'],
            visible = "legendonly",
            opacity = 0,
            name = "Ev. Apertura"
        )
        data.append(vapertura)
        
    except KeyError:
        pass
    
    try:
        vminimo = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['vminimo'],
            visible = "legendonly",
            opacity = 0,
            name = "Ev. Mínimo"
        )
        data.append(vminimo)
    except KeyError:
        pass
    
    try:
        vmaximo = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['vmaximo'],
            visible = "legendonly",
            opacity = 0,
            name = "Ev. Máximo"
        )
        data.append(vmaximo)
    except KeyError:
        pass
    
    try:
        vvolumen = go.Scatter(
            x=df['fecha'], # assign x as the dataframe column 'x'
            y=df['vvolumen'],
            visible = "legendonly",
            opacity = 0,
            name = "Ev. Volumen"
        )
        data.append(vvolumen)
    except KeyError:
        pass
    
    updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list(botones),
    )
    ])
    
    nombre_grafo = 'Registro Financiero de ' + df['ticker'][0][df['ticker'][0].find("/")+1:]
    layout = dict(title=nombre_grafo, showlegend=False,
              updatemenus=updatemenus,
              shapes=lineas_divisorias)
    fig = dict(data=data, layout=layout)
    
    return py.plot(fig, filename=df['ticker'][0].replace('/', "_"))

def visualize_clusters(X, var, color = 'cluster'):
    '''
    Prints with ggplot a visualization of the different clusters.
    '''
    aux = pandas.DataFrame()
    
    aux['fecha'] = X.index
    aux.index = X.index
    
    aux[var] = X[var]
    aux['Cluster'] = X[color]
    
    return ggplot(aes(x='fecha', y=var, color='Cluster'), aux) + geom_point() + xlab(var) + ylab("Valor") + ggtitle("Clustering de la variable \"" + var + "\"") +  theme(axis_text_x  = element_text(color=[0,0,0,0]))

def visualize_segmentation(X, var):
    '''
    Prints with ggplot a visualization of the different segments.
    '''
    aux = pandas.DataFrame(index = X.index)
    
    aux['fecha'] = X.index.values
    aux[var] = X[var]
    aux['Segmento'] = X['segmento'].astype(str)
    
    return ggplot(aes(x="fecha", y=var, color="Segmento"), aux) + geom_point() + xlab("Fecha") + ylab(var) + ggtitle("Segmentacion de la variable \"" + var + "\"") +  theme(axis_text_x  = element_text(color=[0,0,0,0]))
    

def biplot(X, color='cluster'):
    '''
    Prints a biplot with ggplot. Requires color variable: "cluster" in the dataframe.
    '''
    pca = PCA(n_components=2)
    
    res = pca.fit_transform(filter_numerical(X))
    
    df = pandas.DataFrame(res)
    df.columns = ["x", "y"]
    
    if color == 'cluster':
        df['Cluster'] = X[color].values
        color = 'Cluster'
    else:
        c = X[color].values
        c[c=="1"] = "Normal"
        c[c=="-1"] = "Anomalia"
        df['Detectado como:'] = c
        color = 'Detectado como:'
    
    return ggplot(aes("x","y", color=color),df) + geom_point(aes(size=40))

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
    
    
    
    