# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:17:59 2018

@author: Javier Fumanal Idocin
"""
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pam
from sklearn.cluster import KMeans
from sklearn import preprocessing, metrics
from fastdtw import fastdtw

def get_segments_nparray(X, segments):
    resultado = []
    for segmento in segments:
        inicio = segmento[0]
        final = segmento[1]
        
        resultado.append(X[inicio:final,:])
    
    return resultado

def filter_numerical(df):
    '''
    Returns a data frame only with the financial numerical values.
    (apertura, cierre, minimo, maximo, volumen)
    '''
    try: 
        candidatos = ["ticker", "fecha", "cluster", 'anomaly']
        finalistas = []
        for i in candidatos:
            if i in list(df):
                finalistas.append(i)
        return df.drop(finalistas, axis=1)
    except ValueError:
        return df
    
def extract_features(data, f,l, silence2=[], pesos = [1,1,1,1,1,1,1,1,1,1,1,1], fecha=False):
    '''
    Extract features from an interval in a dataset
    '''
    if not fecha:
        rows = data.iloc[f:l,:]
    else:
        rows = data.loc[f:l,:]
    rows = filter_numerical(rows)
    medias = rows.mean(axis=0)
    try:
        medias['vvolumen'] = medias['vvolumen']*0.01
    except KeyError:
        pass
    
        
    if len(silence2) > 0:
        silence = silence2.copy()
        absolutos = medias[["apertura", "maximo", "minimo", "cierre", "volumen", "var"]].drop(silence, errors='ignore',axis=0)
        for i, val in enumerate(silence):
            silence[i] = "v"+val
        tendencias = medias[["vapertura", "vmaximo", "vminimo", "vcierre", "vvolumen", "vvar"]].drop(silence,errors='ignore', axis=0)
    else:
        absolutos = medias[["apertura", "maximo", "minimo", "cierre", "volumen", "var"]]
        tendencias = medias[["vapertura", "vmaximo", "vminimo", "vcierre", "vvolumen", "vvar"]]
        
    return [absolutos * pesos[0:int(len(pesos)/2)], tendencias * pesos[int(len(pesos)/2):]]

def full_clustering(X, segmentos, n_clus = 3, mode="K-Means", silencio=[], pesos = None, normalizar = True):
    '''
    Given a data frame and their segmentation (segments indexes) this function applies
    clustering for each segment and returns the dataframe with the representation of 
    the segments.
    '''
    X.drop('cluster',axis=1,errors='ignore', inplace=True)
    if pesos is None:
         pesos = [1] * (len(list(X._get_numeric_data())) - len(silencio)*2)
            
    if mode == "K-Means":    
         
        segments_df = apply_segmentation(X, segmentos, silencio, pesos)
            
        if normalizar:
            segments_df = minmax_norm(segments_df)
        
        fit = clustering(segments_df, n_clus)
        
        segments_df['cluster'] = fit.labels_
        segments_df['cluster'] = segments_df['cluster'].astype(str)
        
    else:
        X_num = filter_numerical(X)
        
        if normalizar:
            X_num = minmax_norm(X_num)
            
        X_num = filter_silence(X_num, silencio)
        X_np = np.array(X_num)
        
        X_segments = get_segments_nparray(X_np, segmentos)
            
        _, best_choice, best_res =  pam.kmedoids(X_segments, n_clus)

        fit = FTWFit(best_choice, filter_numerical(X))
        
        segments_df = apply_segmentation(X, segmentos, silencio, pesos)
        segments_df = add_clustering_segments(segments_df, best_choice, best_res)
        if normalizar:
            segments_df = minmax_norm(segments_df)
        
    return segments_df, fit

def add_clustering_segments(segmentos, clusters, asociaciones):
    '''
    Cluster each point individually according to each segment cluster.
    '''
    segmentos['cluster'] = 0
    
    for cluster in clusters:
        for i_segment in asociaciones[cluster]:
            segmentos['cluster'][i_segment] = cluster
    
    segmentos['cluster'] = segmentos['cluster'].astype(str)
    return segmentos

def filter_silence(X, silencios):
    '''
    '''
    return X.drop(silencios, axis=1)

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
    try:
        H = sum(ujd) / (sum(ujd) + sum(wjd))
    except ZeroDivisionError:
        H = 0
    
    if np.isnan(H):
        H = 0
 
    return H
    
def apply_segmentation(X, segmentos, silencio=[], pesos = [1,1,1,1,1,1, 1,1,1,1,1,1], fecha=False):
    '''
    Returns the features of each segment in the dataset.
    '''
    segments_df = pd.DataFrame()
    for seg in segmentos:
        inicio = seg[0]
        final = seg[1]
        
        f = extract_features(X,inicio, final, silencio, pesos, fecha)
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
    res = filter_numerical(X)
    res['cluster'] = X['cluster']
    return metrics.silhouette_score(res, res['cluster'], metric='sqeuclidean')


class FTWFit:
    
    def __init__(self, keys, values):
        self.centroides = {}
        for i in keys:
            self.centroides[i] = values.iloc[i]
            
    def predict(self, point):
        best = np.inf
        for key, value in self.centroides.items():
            tmp = fastdtw(point, value)[0]
            
            if best > tmp:
                best = tmp
                res = key
            
        return res
        