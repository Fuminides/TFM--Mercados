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
from sklearn import preprocessing, metrics

def filter_numerical(df):
    '''
    Returns a data frame only with the financial numerical values.
    (apertura, cierre, minimo, maximo, volumen)
    '''
    try: 
        return df.drop(["ticker", "fecha"], axis=1)
    except ValueError:
        return df
    
def extract_features(data, f,l, silence2=[], pesos = [1,1,1,1,1,1,1,1,1,1,1,1]):
    '''
    Extract features from an interval in a dataset
    '''
    rows = data.iloc[f:l,:]
    rows = filter_numerical(rows)
    medias = rows.mean(axis=0)
    try:
        medias['vvolumen'] = medias['vvolumen']*0.01
    except KeyError:
        pass
    
    if len(silence2) > 0:
        silence = silence2.copy()
        absolutos = medias[["apertura", "maximo", "minimo", "cierre", "volumen", "var"]].drop(silence, axis=0)
        for i, val in enumerate(silence):
            silence[i] = "v"+val
        tendencias = medias[["vapertura", "vmaximo", "vminimo", "vcierre", "vvolumen", "vvar"]].drop(silence, axis=0)
    else:
        absolutos = medias[["apertura", "maximo", "minimo", "cierre", "volumen", "var"]]
        tendencias = medias[["vapertura", "vmaximo", "vminimo", "vcierre", "vvolumen", "vvar"]]
    
    return [absolutos * pesos[0:int(len(pesos)/2)], tendencias * pesos[int(len(pesos)/2)]]

def full_clustering(X, segmentos, n_clus = 3, silencio=[], pesos = [1,1,1,1,1,1, 1,1,1,1,1,1]):
    '''
    '''
    segments_df = apply_segmentation(X, segmentos, silencio, pesos)
    names = segments_df.columns.values
    segments_df = minmax_norm(segments_df)
    segments_df.columns = names
    
    fit = clustering(segments_df, n_clus)
    
    segments_df['cluster'] = fit.labels_
    
    return segments_df

def cluster_points(X, segmentos, clusters):
    '''
    '''
    rows = X.shape[0]
    res = [0]*rows
    for index, seg in enumerate(segmentos):
        inicio = seg[0]
        fin = seg[1]
        clus = clusters[index]
        res[inicio:fin] = [clus]*(fin-inicio)
        
    X['cluster'] = res
    
    return X
    
def clustering(X, n_clusters=3):
    '''
    Applies K-Means to a dataset given the number of clusters
    '''
    return KMeans(n_clusters=2, random_state=0).fit(X)
    
def hopkins_statistic(X, m=10):
    '''
    Return the Hopkins statistic for clustering tendency in a data set.
    '''
    n = np.shape(X)[0]
    Y = random.sample(range(0, n, 1), m)
    
    if type(X)==pd.core.frame.DataFrame:
        nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
    else:
         nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    
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
    
def apply_segmentation(X, segmentos, silencio=[], pesos = [1,1,1,1,1,1, 1,1,1,1,1,1]):
    '''
    Returns the features of each segment in the dataset.
    '''
    segments_df = pd.DataFrame()
    for seg in segmentos:
        inicio = seg[0]
        final = seg[1]
        
        f = extract_features(X,inicio, final, silencio, pesos)
        segments_df = segments_df.append(f[0].append(f[1]), ignore_index = True)
        
    return segments_df

def apply_clustering(X, segmentos, clusters, asociaciones):
    '''
    '''
    X['cluster'] = 0
    for cluster in clusters:
        for i_segment in asociaciones[cluster]:
            seg = segmentos[i_segment]
            inicio = seg[0]
            final = seg[1]
            
            X['cluster'].iloc[inicio:final] = cluster
        
    return X
        
def minmax_norm(df):
    '''
    '''
    df = filter_numerical(df)
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)

def sil_metric(X)     :
    '''
    '''
    return metrics.silhouette_score(X, X['cluster'], metric='sqeuclidean')