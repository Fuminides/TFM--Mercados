# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:54:58 2018

@author: Javier Fumanal Idocin
"""
import numpy as np
import orangecontrib.associate.fpgrowth as fp
from clustering import clustering, filter_numerical
from sklearn.ensemble import IsolationForest

from eventregistry import EventRegistry, QueryArticlesIter

def extremos_incertidumbre(fit, datos, cluster):
    '''
    Devuelve la mayor y menor distancia de un conjunto de datos a un fit, dada una funcion
    de distancia.
    '''
    maximo = 0
    minimo = np.Inf
    
    for dato in range(datos.shape[0]):
        dist = distancia(fit, datos.iloc[dato].values.reshape(1,-1), cluster)
        
        if dist > maximo:
            maximo = dist
        elif dist < minimo:
            minimo = dist
    
    return [maximo, minimo]

def calcular_incertidumbre(fit, dato, confianzas, cluster):
    '''
    Calculate the confidence in a prediction.
    '''
    minimo = confianzas[1]
    maximo = confianzas[0]
    
    dist = distancia(fit, dato, cluster)
    
    confianza = (dist - minimo) / maximo
    
    if(confianza < 0):
        return 0
    else:
        return confianza
        
def distancia(fit, dato, cluster):
    '''
    Calcultes distances from points to clusters.
    '''
    return fit.transform(dato)[0][cluster]

def classify(datos, fit):
    '''
    Classifies data.
    '''
    return fit.predict(datos)

def tasa_aceptabilidad(incertidumbres):
    '''
    Gives the percentage of acceptable classifications made.
    '''
    incertidumbres2 = np.array(incertidumbres.copy())
    incertidumbres2[np.array(incertidumbres2) < 1.0] = 1.0 
    incertidumbres2[np.array(incertidumbres2) > 1.0] = 0.0
    
    return np.mean(incertidumbres2)

def transacciones_profundidad(X, profundidad = 1):
    '''
    '''
    clusters = np.array(X['cluster'])
    rows = len(clusters) - profundidad
    cols = len(np.unique(X.cluster))
    holder = np.zeros([rows, cols])
    holder.fill('0')
    
    for i in range(profundidad, rows):
        holder[i-profundidad,clusters[i]] = 1
        
        for j in range(i-profundidad,i):
            holder[i - profundidad,clusters[j]] = 1
    
    return holder

def rules_extractor(X, profundidades=range(4), metric = 0.3):
    res = {}
    
    for i in profundidades:
        T = transacciones_profundidad(X,i)
        
        itemsets = dict(fp.frequent_itemsets(T, metric))
        rules = [(P, Q, supp, conf) for P, Q, supp, conf in fp.association_rules(itemsets, metric)]
        
        res[i] = (itemsets, rules)
 
    return res      

def anomaly_detection(X, name = 'anomaly'):
    pr = IsolationForest()
    pr.fit(filter_numerical(X))
    x = pr.predict(filter_numerical(X))
    X[name] = x
    X[name] = X[name].astype(str)
        
    
def noticias(theme, dates):
    er = EventRegistry("c4b6b663-d180-4f4c-8163-4c45fbc7cbd7")
    q = QueryArticlesIter(conceptUri = theme, dateStart = dates[0], dateEnd=dates[1]  )
    for art in q.execQuery(er, sortBy = "date"):
        print(art)
    
def conjunto_segmentados(segmentados, n=3):
    import pandas as pd
    segments_df=pd.DataFrame()
    lens = []
    inicio=0
    
    for conj in segmentados:
        segments_df = segments_df.append(conj)
        lens.append([inicio,conj.shape[0]])
        inicio = conj.shape[0]
    
    fit = clustering(segments_df, n)
    segments_df['cluster'] = fit.labels_
    segments_df['cluster'] = segments_df['cluster'].astype(str)
    
    for x in range(len(lens)):
        segmentados[x] = segments_df.iloc[lens[x]]
    confidences = []
    for i in range(n):
        confidences.append(extremos_incertidumbre(fit, filter_numerical(segments_df), i))
        
    return segmentados, fit, confidences, segments_df

def clustering_df(X, fit, confidences=None):
    clusters = []
    confs = []
    X.drop('cluster', axis=1,inplace=True, errors='ignore')
    
    for i in range(X.shape[0]):
        pred = classify(X.iloc[i].values.reshape(1,-1), fit)
        dis = distancia(fit, X.iloc[i].values.reshape(1,-1), pred)[0]

        if not(confidences is None):
            conf = (dis - confidences[pred[0]][1]) / confidences[pred[0]][0]
            confs.append(conf)
            
        clusters.append(pred[0])
      
    X['cluster'] = clusters
    
    return confs
    
        
        
        
    
    