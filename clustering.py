# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:17:59 2018

@author: javi-
"""

import numpy as np
import random

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from math import factorial


def normalize_data_frame(df, norm="mean"):
    '''
    Normalize data columns in the data frame. Returns also the normalization parameters.
    '''
    if norm == "mean":
        medias = df.mean()
        desviaciones = df.std()
        normalized_df=(df-df.mean())/df.std()

        return [normalized_df, medias, desviaciones]
    elif norm == "min-max":
        mini = df.min()
        maxi = df.max()
        normalized_df=(df-mini)/(maxi-mini)
        
        return [normalized_df, mini, maxi]


def regresion_polinomial(X,y, degree=2):
    '''
    Adjust polynomic model to data. (Might be used to smooth data)
    '''
    model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
    model.fit(X, y)
    
    return model

def savitzky_golay(y, window_size=51, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')

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
 
def extract_features(data, segments):
    '''
    Extract features from an interval in a dataset
    '''
    
def similarity(f1,f2):
    '''
    Computes the average of the differences between two vectors.
    '''
    return np.mean(f1-f2)

def valid_segment(data, intervalo, similarity_function, threshold, montecarlo=4):
    '''
    Returns true only if the designated segment is considered to be valid.
    It samples some subsegments and comparates their features with a distance function.
    '''
    first = intervalo[0]
    last = intervalo[1]
    segments=[]
    
    for i in range(montecarlo):
        r1 = random.randint(first+1, last)
        r2 = random.randint(first+1, last)
        
        if r2>r1:
            segments.append([r1,r2])
        elif r1>r2:
            segments.append([r2,r1])
    
    features1 = extract_features(data, segment1)
    features2 = extract_features(data, segment2)
    
    return similarity_function(features1, features2) < threshold
    
def segmentate(intervalo, data, similar_function, threshold, montecarlo=4):
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
    
    if valid_segment(data, intervalo, similar_function, threshold, montecarlo):
        #If this is a valid segment, it finishes
        intervals.append(intervalo)
    else:
        last_check = first
        while(last_check != last):
            #We'll keep going until the whole interval is segmented.     
            success = False     
            while not success:
                new_interval = [last_check, random.randint(last_check+1, last)]
                success = valid_segment(data, new_interval, similar_function, threshold, montecarlo)
                
                if success:
                    #We have found a valid segment, we append it to the list of segments
                    intervals.append(new_interval)
                    last_check = new_interval[1]
                    
    
    return intervals
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    