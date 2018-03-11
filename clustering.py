# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:17:59 2018

@author: Javier Fumanal Idocin
"""

import numpy as np
import random
import progressbar

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats


def locate_local_minmax(series):
    '''
    Return the index for the local max and min points located in the series.
    '''
    local = []
    
    for actual in range(len(series)-2):
        segmento1 = np.sign(series[actual+1] - series[actual])
        segmento2 = np.sign(series[actual+2] - series[actual+1])
        
        if segmento1 != segmento2:
            #We've changed our derivate, then we are at local min/max            
           local.append(actual+1)
           
    return local
                    
                    
def environment_filter(maximals, series, size=10):
    '''
    Filter local mins and max based on their environment.
    '''
    keep = []
    for x in maximals:
        keeps = True
        if x > 0 and x < len(series)-1:
            segmento1 = np.sign(series[x] - series[x-1])
            segmento2 = np.sign(series[x+1] - series[x])
            derecha = x + size
            izquierda = x - size + 1
            
            if(derecha > len(series)):
                derecha = len(series)
            if(izquierda < 0):
                izquierda = 0
            series = np.array(series)
            entorno = series[np.arange(izquierda,derecha)]
            if segmento1 > segmento2:
                #Local max
                if np.max(entorno) > series[x]:
                    keeps = False
            else:
                #Local min
                if np.min(entorno) < series[x]:
                    keeps = False
        
        if keeps:
            keep.append(x)
    
    return keep
 
def extract_features(data, f,l):
    '''
    Extract features from an interval in a dataset
    '''
    rows = data.iloc[f:l,:]
    rows = filter_numerical(rows)
    medias = rows.mean(axis=0)
    
    absolutos = medias[["apertura", "maximo", "minimo", "cierre", "volumen", "var"]]
    tendencias = medias[["vapertura", "vmax", "vmin", "vcierre", "vvolumen"]]
    
    return [absolutos, tendencias]
    
def distance(f1,f2):
    '''
    Computes the differences between two segments.
    '''
    abs1 = f1[0]
    abs2 = f2[0]
    
    ten1 = f1[1]
    ten2 = f2[1]
    
    dabs = (np.abs(abs1 - abs2) / abs2)
    dabs[dabs == np.inf] = 0
    dabs[dabs == -np.inf] = 0
    dabs = np.max(dabs)
    dten = (np.abs(ten1 - ten2) / ten2)
    dten[dten == np.inf] = 0
    dten[dten== -np.inf] = 0
    dten = np.max(dten)
    
    return np.max([dabs, dten])

def filter_numerical(df):
    '''
    Returns a data frame only with the financial numerical values.
    (apertura, cierre, minimo, maximo, volumen)
    '''
    return df.drop(["ticker", "fecha"], axis=1)

def valid_segment(data, intervalo, distance, threshold, montecarlo=4):
    '''
    Returns true only if the designated segment is considered to be valid.
    It samples some subsegments and comparates their features with a distance function.
    '''
    first = intervalo[0]
    last = intervalo[1]
    
    if last-first==1:
        return True
    
    for i in range(montecarlo):
        r11 = random.randint(first+1, last)
        r21 = random.randint(first+1, last)
        
        while r11==r21:
            r11 = random.randint(first+1, last)
            r21 = random.randint(first+1, last)
            
        if r21>r11:
            features1 = extract_features(data, r11,r21)
        elif r11>r21:
            features1 = extract_features(data, r21,r11)
        
        r12 = random.randint(first+1, last)
        r22 = random.randint(first+1, last)
        
        while r12==r22:
            r12 = random.randint(first+1, last)
            r22 = random.randint(first+1, last)
        
        if r22>r12:
           features2 = extract_features(data, r12,r22)
        elif r12>r22:
           features2 = extract_features(data, r22,r12)
               
        good = distance(features1, features2) < threshold

        if not good:
            return False
    
    return True
    
def segmentate(intervalo, data, distance_function, threshold, montecarlo=2):
    '''
    Given set of data, it divides it into segments according to a distance function.
    It uses a interval of numbers of the original data. It is recommended for small parts
    of the original data.
    
    Returns the list of indexes of the segments.
    
    In case you need a segmentation function for a big amount of data use locate_minmax()
    (Requires smoothing)
    '''
    first = intervalo[0]
    last = intervalo[1]
    intervals = []
    
    if valid_segment(data, intervalo, distance_function, threshold, montecarlo):
        #If this is a valid segment, it finishes
        intervals.append(intervalo)
    else:
        last_check = first
        while(last_check != last):
            #We'll keep going until the whole interval is segmented.     
            success = False     
            while not success:
                new_interval = [last_check, random.randint(last_check+1, last)]
                success = valid_segment(data, new_interval, distance_function, threshold, montecarlo)
                
                if success:
                    #We have found a valid segment, we append it to the list of segments
                    intervals.append(new_interval)
                    last_check = new_interval[1]
                
    return intervals

def join_segments(data, o_segments, distance, threshold):
    '''
    Joins segments which are very similar and adjacent. That is, segments which are
    in reality just one.
    '''
    res = []
    segments = o_segments.copy()
    for i in range(len(segments)-1):
        first = segments[i][0]
        last = segments[i][1]
        
        f1 = extract_features(data, first, last)
        
        first2 = segments[i+1][0]
        last2 = segments[i+1][1]
        
        f2 = extract_features(data, first2, last2)
        
        same = distance(f1,f2) < threshold
        
        if same:
            segments[i+1][0] = first
            segments[i][1] = last2
            
        else:
            res.append([first, last])
            
    res.append(segments[-1])
        
    return res

def segmentate_data_frame(df, montecarlo = 8, trh = 0.5):
    '''
    Given a financial data frame, it gets segmentated according to standard parameters.
    '''
    cierre = df['cierre']
    maxmin = locate_local_minmax(cierre)
    maximals = environment_filter(maxmin, cierre, size=25)
    inicio = 0
    res = []
    bar = progressbar.ProgressBar()
    bar.max_value = len(maximals)
    bar.min_value = 0
    bar.update(0)
    index = 0
    
    for i in maximals:
        rango = [inicio, i]
        subsegmentos = segmentate(rango, df, distance, 0.5, montecarlo)
        res.append(subsegmentos)
        index += 1
        bar.update(index)
        inicio = i
    
    bar.finish()
    
    return join_segments(df, res, distance, trh)   
        
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    