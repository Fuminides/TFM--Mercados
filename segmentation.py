# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:17:59 2018

@author: Javier Fumanal Idocin
"""
import numpy as np
import random
import progressbar

from joblib import delayed, Parallel, cpu_count

def _num_after_point(x):
    '''
    Given a float number x, it returns the number of zeroes between the
    dot and the first non zero decimal number.
    
    x - a float number less than 1.
    '''
    for i in range(10):
        comp = 0.1**i
        if comp <= x:
            return i
        
    return 10
        

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
    
def distance(f1,f2, vector_importancias, penalizacion_orden=3):
    '''
    Computes the differences between two segments.
    '''
    abs1 = f1[0]
    abs2 = f2[0]
    
    ten1 = f1[1]
    ten2 = f2[1]
    
    dabs = (np.abs(abs1 - abs2) / abs2)
    
    try:
        with np.errstate(divide='ignore'):
            pens = (1 + np.e**-(np.log10(abs(abs1-abs2))-penalizacion_orden))
        pens[pens.index != 'volumen'] = 1
        dabs = dabs / pens
            
    except KeyError:
        dabs=dabs
    
    dabs[dabs == np.inf] = 0
    dabs[dabs == -np.inf] = 0
    dabs = np.max(dabs * vector_importancias[0:int(len(vector_importancias)/2)])
    with np.errstate(divide='ignore'):
        dten = np.log10(np.abs(ten1 - ten2))+3 #Es * 1000 en el log
        
    dten[dten == np.inf] = 0
    dten[dten== -np.inf] = 0
    dten2 = dten * vector_importancias[int(len(vector_importancias)/2):]
    dten = np.max(dten2)
    
    return np.max([dabs, dten])

def interpretable_distance(f1,f2, silence2=[], pen_orden=3, vector_importancias = [1,1,1,1,1,1, 1,1,1,1,1,1]):
    '''
    Computes the differences between two segments, and gives the feature that causes the max distance.
    '''
    abs1 = f1[0]
    abs2 = f2[0]
    
    ten1 = f1[1]
    ten2 = f2[1]
    absolutos = ["apertura", "maximo", "minimo", "cierre", "volumen", "var"]
    tendencias = ["vapertura", "vmaximo", "vminimo", "vcierre", "vvolumen", "vvar"]
    
    if len(silence2)>0:
        silence=silence2.copy()
        absolutos = [x for x in absolutos if x not in silence]
        for i, val in enumerate(silence):
            silence[i] = "v"+val
        
        tendencias = [x for x in tendencias if x not in silence]
        
    max_distancia = 0.0
    nombre = "Ninguno"
    
    for index, name in enumerate(absolutos):
        val1 = abs1[index] * vector_importancias[index]
        val2 = abs2[index] * vector_importancias[index]
        dif = (np.abs(val1 - val2) / val2)
        
        if name == 'volumen':
            pens = (1 + np.e**-(np.log10( abs(val1-val2) ) - pen_orden))
            dif = dif / pens            

        if (dif == np.inf) or (dif == -np.inf):
            dif=0
            
        dif = dif#*_num_after_point(dif)
        if max_distancia < dif:
            max_distancia = dif
            nombre = name
    
    for index, name in enumerate(tendencias):
        val1 = ten1[index] * vector_importancias[index]
        val2 = ten2[index] * vector_importancias[index]
        dif = np.log10(np.abs(val1 - val2))+3

        if (dif == np.inf) or (dif == -np.inf):
            dif=0
        dif = dif#*_num_after_point(dif)
        if max_distancia < dif:
            max_distancia = dif
            nombre = name
    
    return [max_distancia, nombre]

def filter_numerical(df):
    '''
    Returns a data frame only with the financial numerical values.
    (apertura, cierre, minimo, maximo, volumen)
    '''
    return df.drop(["ticker", "fecha"], axis=1)

def valid_segment(data, intervalo, distance, threshold, montecarlo=4, silence=[], penalizacion_orden=3, vector_importancias = [1,1,1,1,1,1,1,1,1,1,1,1]):
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
            features1 = extract_features(data, r11,r21, silence)
        elif r11>r21:
            features1 = extract_features(data, r21,r11, silence)
        
        r12 = random.randint(first+1, last)
        r22 = random.randint(first+1, last)
        
        while r12==r22:
            r12 = random.randint(first+1, last)
            r22 = random.randint(first+1, last)
        
        if r22>r12:
           features2 = extract_features(data, r12,r22, silence)
        elif r12>r22:
           features2 = extract_features(data, r22,r12, silence)
               
        good = distance(features1, features2, vector_importancias, penalizacion_orden) < threshold

        if not good:
            return False
    
    return True
    
def segmentate(intervalo, data, distance_function, threshold, montecarlo=2, silence=[], pen = 3, vector_importancias = [1,1,1,1,1,1,1,1,1,1,1,1]):
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
    if valid_segment(data=data, intervalo= intervalo, distance=distance_function, threshold=threshold, montecarlo=montecarlo, silence=silence, penalizacion_orden = pen, vector_importancias = vector_importancias):
        #If this is a valid segment, it finishes
        intervals.append(intervalo)
    else:
        last_check = first
        while(last_check != last):
            #We'll keep going until the whole interval is segmented.     
            success = False     
            while not success:
                new_interval = [last_check, random.randint(last_check+1, last)]
                success = valid_segment(data=data, intervalo=new_interval, distance=distance_function, threshold=threshold, montecarlo=montecarlo,silence=silence, penalizacion_orden=pen, vector_importancias=vector_importancias)
                
                if success:
                    #We have found a valid segment, we append it to the list of segments
                    intervals.append(new_interval)
                    last_check = new_interval[1]
                
    return intervals

def join_segments(data, o_segments, distance, threshold, minimum_size=5, silence=[], penalizacion_orden=3, vector_importancias = [1,1,1,1,1,1,1,1,1,1,1,1]):
    '''
    Joins segments which are very similar and adjacent. That is, segments which are
    in reality just one.
    '''
    res = []
    segments = o_segments.copy()
    explanations = []
    for i in range(len(segments)-1):
        first = segments[i][0]
        last = segments[i][1]
        
        f1 = extract_features(data, first, last, silence)
        
        first2 = segments[i+1][0]
        last2 = segments[i+1][1]
        
        f2 = extract_features(data, first2, last2, silence)
        
        if (last-first+1 < minimum_size) or (last2-first2+1 < minimum_size):
            same = True
        else:
            cut = distance(f1,f2, silence, penalizacion_orden, vector_importancias)
            same = cut[0] < threshold
        
        if same:
            segments[i+1][0] = first
            segments[i][1] = last2
            
        else:
            explanations.append(cut)
            res.append([first, last])
            
    res.append(segments[-1])
        
    return res, explanations

def segmentate_data_frame(df, montecarlo = 8, trh = 0.5, min_size=3, silence = [], penalizacion_orden = 3, verbose = False, vector_importancias = None):
    '''
    Given a financial data frame, it gets segmentated according to standard parameters.
    '''
    cierre = df['cierre']
    maxmin = locate_local_minmax(cierre)
    maximals = environment_filter(maxmin, cierre, size=25)
    inicio = 0
    res = []
    if verbose:
        bar = progressbar.ProgressBar()
        bar.max_value = len(maximals)
        bar.min_value = 0
        bar.update(0)
    index = 0
    
    if vector_importancias is None:
        vector_importancias = [1] * (len(list(df._get_numeric_data())) - len(silence)*2)
        
    for i in maximals:
        rango = [inicio, i]
        subsegmentos = segmentate(intervalo=rango, data=df, distance_function=distance, threshold=0.5, montecarlo=montecarlo, silence=silence, pen=penalizacion_orden,vector_importancias=vector_importancias)
        res = res + subsegmentos #res.append(subsegmentos)
        index += 1
        
        if verbose:
            bar.update(index)
            
        inicio = i
    if verbose:
        bar.finish()

    return join_segments(df, res, interpretable_distance, trh, min_size, silence, penalizacion_orden= penalizacion_orden,vector_importancias = vector_importancias)

def _segmentate_subrange(i, maximals, df, distance, montecarlo, trh):
    if i==0:
        inicio=0
    else:
        inicio = maximals[i-1]
        
    final = maximals[i]
    
    rango = [inicio, final]
    res = segmentate(rango, df, distance, trh, montecarlo)
    
    return res
    
def parallel_segmentate_data_frame(df, montecarlo = 8, trh = 0.5, min_size=3, silence=[],penalizacion_orden=3, verb = 50, vector_importancias = [1,1,1,1,1,1,1,1,1,1,1,1]):
    '''
    Given a financial data frame, it gets segmentated according to standard parameters.
    '''
    cierre = df['cierre']
    maxmin = locate_local_minmax(cierre)
    maximals = environment_filter(maxmin, cierre, size=25)
    
    res = []
    
    iters = len(maximals)
    ncpu = cpu_count()
    segs = Parallel(n_jobs=ncpu, verbose=verb)(delayed(_segmentate_subrange)(i,maximals,df,distance, montecarlo,trh,penalizacion_orden) for i in range(iters))
    res = []
    for i in segs:
        res = res + i
        
    return join_segments(df, res, interpretable_distance, trh, min_size, silence , vector_importancias)

def get_segments(X, segments):
    '''
    '''
    resultado = []
    for segmento in segments:
        inicio = segmento[0]
        final = segmento[1]
        
        resultado.append(X.iloc[inicio:final,:].drop(['fecha','ticker'], axis=1).values)
    
    return resultado

def get_segments_nparray(X, segments):
    resultado = []
    for segmento in segments:
        inicio = segmento[0]
        final = segmento[1]
        
        resultado.append(X[inicio:final,:])
    
    return resultado
    
def get_years(data):
    '''
    Returns and array with yearly-separated data.
    '''
    grupos = []
    data_groups = data.groupby(data.index.year)
    
    for i in data_groups.groups.keys():
        grupos.append(data_groups.get_group(i))
    
    return grupos
    