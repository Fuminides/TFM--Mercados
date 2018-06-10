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
        candidatos = ["ticker", "fecha", "cluster"]
        finalistas = []
        for i in candidatos:
            if i in list(df):
                finalistas.append(i)
        return df.drop(finalistas, axis=1)
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
    
    return [absolutos * pesos[0:int(len(pesos)/2)], tendencias * pesos[int(len(pesos)/2):]]

def full_clustering(X, segmentos, n_clus = 3, silencio=[], pesos = None, normalizar = False):
    '''
    Given a data frame and their segmentation (segments indexes) this function applies
    clustering for each segment and returns the dataframe with the representation of 
    the segments.
    '''
    if pesos is None:
        pesos = [1] * (len(list(X._get_numeric_data())) - len(silencio)*2)
        
    segments_df = apply_segmentation(X, segmentos, silencio, pesos)
        
    if normalizar:
        segments_df = minmax_norm(segments_df)
    
    fit = clustering(segments_df, n_clus)
    
    segments_df['cluster'] = fit.labels_
    segments_df['cluster'] = segments_df['cluster'].astype(str)
    
    return segments_df, fit

def cluster_points(X, segmentos, clusters):
    '''
    Given a data frame of points, it's segmentation and it's segments cluster, it
    clusters each point individually.
    '''
    rows = X.shape[0]
    res = [0]*rows
    for index, seg in enumerate(segmentos):
        inicio = seg[0]
        fin = seg[1]
        clus = clusters[index]
        res[inicio:fin] = [clus]*(fin-inicio)
        
    X['cluster'] = res
    X['cluster'] = X['cluster'].astype(str)
    
    
def clustering(X, n_clusters=3):
    '''
    Applies K-Means to a dataset given the number of clusters
    '''
    return KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
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

def add_clustering(X, segmentos, clusters, asociaciones):
    '''
    Cluster each point individually according to each segment cluster.
    '''
    X['cluster'] = 0
    
    for cluster in clusters:
        for i_segment in asociaciones[cluster]:
            seg = segmentos[i_segment]
            inicio = seg[0]
            final = seg[1]
            
            X['cluster'].iloc[inicio:final] = cluster
    
    X['cluster'] = X['cluster'].astype(str)
    return X

def add_segmentation(X, segmentos):
    '''
    Add each segmentation to each point individually.
    '''
    X['segmento'] = 0
    for index, seg in enumerate(segmentos):
        inicio = seg[0]
        final = seg[1]
        
        X['segmento'].iloc[inicio:final] = index
    
    ultimo = segmentos[-1][1]
    X['segmento'].iloc[ultimo:X.shape[0]] = index+1
        
    return X
        
def minmax_norm(df_0):
    '''
    Normalize data with the minmax norm.
    '''
    df = filter_numerical(df_0)
    names = list(df)
    x = df.values
    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    holder = pd.DataFrame(x_scaled)
    
    holder.index = df_0.index
    holder.columns = [names]
    
    for i in names:
        df_0[i] = holder[i]
    
    return df_0

def sil_metric(X):
    '''
    Return the sillhoutte coefficent of a data frame that has a 'cluster' column.
    '''
    return metrics.silhouette_score(X, X['cluster'], metric='sqeuclidean')

