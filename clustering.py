# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:17:59 2018

@author: Javier Fumanal Idocin
"""
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from sklearn.cluster import KMeans

def filter_numerical(df):
    '''
    Returns a data frame only with the financial numerical values.
    (apertura, cierre, minimo, maximo, volumen)
    '''
    return df.drop(["ticker", "fecha"], axis=1)

def extract_features(data, f,l, silence2=[]):
    '''
    Extract features from an interval in a dataset
    '''
    rows = data.iloc[f:l,:]
    rows = filter_numerical(rows)
    medias = rows.mean(axis=0)
    medias['vvolumen'] = medias['vvolumen']*0.01
    if len(silence2) > 0:
        silence = silence2.copy()
        absolutos = medias[["apertura", "maximo", "minimo", "cierre", "volumen", "var"]].drop(silence, axis=0)
        for i, val in enumerate(silence):
            silence[i] = "v"+val
        tendencias = medias[["vapertura", "vmaximo", "vminimo", "vcierre", "vvolumen", "vvar"]].drop(silence, axis=0)
    else:
        absolutos = medias[["apertura", "maximo", "minimo", "cierre", "volumen", "var"]]
        tendencias = medias[["vapertura", "vmaximo", "vminimo", "vcierre", "vvolumen", "vvar"]]
    
    return [absolutos, tendencias]

def full_clustering(X, segmentos, n_clus = 3, silencio=[]):
    segments_df = pd.DataFrame()
    for seg in segmentos:
        inicio = seg[0]
        final = seg[1]
        
        f = extract_features(X,inicio, final, silencio)
        segments_df = segments_df.append(f)
    
    fit = clustering(segments_df, n_clus)
    
    segments_df['cluster'] = fit.labels_
    
    return segments_df
    
    
def clustering(X, n_clusters=3):
    return KMeans(n_clusters=2, random_state=0).fit(X)
    
def hopkins_statistic(X, m=10):
    '''
    Return the Hopkins statistic for clustering tendency given a data set.
    '''
    n = np.shape(X)[0]
    Y = random.sample(range(0, n, 1), m)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
    
    ujd = []
    wjd = []
    
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(random.uniform(np.amin(X,axis=0),np.amax(X,axis=0)).values.reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[Y[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    
    if np.isnan(H):
        H = 0
 
    return H
    
    
    