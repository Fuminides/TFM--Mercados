# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:45:12 2018

@author: javi-
"""
import pandas as pd
import numpy as np
from numbers import Number

from scipy.signal import butter, filtfilt
from statsmodels.nonparametric.smoothers_lowess import lowess
from math import factorial

def full_preprocess(df):
    '''
    Fills empty data, crops outliers using bollinger bands and creates new variables in
    the data frame.
    '''
    df = fill_dates(df)
    impute_missing(df)
    for i in ['apertura', 'cierre', 'minimo', 'maximo', 'volumen']:    
        crop_outliers(df, i)
    filter_data_frame(df)
    augment_data(df)
    
    return df
    
    
    
def _number_to_date(number_date):
    return str(number_date)[:10]

def _dates_to_string(colum_dates):
    res = []
    for i in range(len(colum_dates)):
        res.append(_number_to_date(colum_dates[i]))
    
    return res

def fill_dates(df):
    '''
    Fill the gaps in missing/non working days with closer existing value.
    '''
    fechas_o = df.index.values
    inicio = fechas_o[0]
    final = fechas_o[-1]
    fechas = pd.date_range(inicio, final)
    tamano = len(df.loc[inicio,])
    ticker = df.loc[inicio,"ticker"]
    for fecha in fechas:
        try:
            df.loc[fecha, ]
        except KeyError:
            df.loc[fecha] = [np.nan]*tamano
            df.loc[fecha, 'fecha'] = _number_to_date(fecha)
            df.loc[fecha, 'ticker'] = ticker
    
    df = df.sort_index()
    return df
    
    
def impute_missing(df):
    '''
    Impute missing values in a data frame.
    '''
    for columna in range(len(df.columns)):
        for i in range(len(df.iloc[:,columna])):
            dato = df.iloc[i, columna]
            
            if pd.isnull(dato):
                if i>0 and not pd.isnull(df.iloc[i-1, columna]):
                    df.iloc[i, columna] = df.iloc[i-1,columna]
                elif i+1<len(df.iloc[:,columna]) and not pd.isnull(df.iloc[i+1, columna]):
                    df.iloc[i, columna] = df.iloc[i+1,columna]
                    
    return df 

def _Bolinger_Bands(stock_price, window_size=21, num_of_std=2):

    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)

    return rolling_mean, upper_band, lower_band

def detect_finantial_outliers(df, variable, window_size=21):
    '''
    Detects values outside the bolinger band.
    '''
    mean, upper, lower = _Bolinger_Bands(df[variable], window_size)
    minors = []
    majors = []
    for i in range(len(df[variable])):
        if df[variable][i]> upper[i]:
            majors.append(i)
        elif df[variable][i] < lower[i]:
            minors.append(i)
            
    return [minors, majors]

def crop_outliers(df, variable, window_size=21):
    '''
    If any value is outside the bolinger band, it gets erased.
    '''
    mean, upper, lower = _Bolinger_Bands(df[variable], window_size)
    minors, majors = detect_finantial_outliers(df,variable, window_size)
    for i in minors:
        df[variable][i] = lower[i]
    
    for i in majors:
        df[variable][i] = upper[i]
        
    return df

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

def augment_data(stock_values, reference_index=None):
    '''
    Given a financial data_frame it creates the new derived atributes:
        -Variability: difference between min and max value that day.
        -Percentaje variability: variability divided by the stock price
        -Ascend: true if the value today is more valuable than yesterday.
        -Evolution: the diference between yesterday price and today's
        -Index: difference between than it's reference index.
        -Volume change: the difference between the number of stock sold the
                        day before and the next one.
        -WIP
    '''
    variab = np.zeros(stock_values.shape[0])
    vvariab = np.zeros(stock_values.shape[0])
    vcierre = np.zeros(stock_values.shape[0])
    vapertura = np.zeros(stock_values.shape[0])
    vvolume = np.zeros(stock_values.shape[0])
    vmax = np.zeros(stock_values.shape[0])
    vmin = np.zeros(stock_values.shape[0])

    if reference_index != None:
        good = np.zeros(stock_values.shape[0])

    yesterday = stock_values.iloc[0]
    
    for i in np.arange(stock_values.shape[0]):
        row = stock_values.iloc[i]

        if yesterday['cierre'] == 0:
            vcierre[i] = 0
        else:
            vcierre[i] = (row['cierre'] - yesterday['cierre']) / yesterday['cierre']
            
        if yesterday['apertura'] == 0:
            vapertura[i] = 0
        else:
            vapertura[i] = (row['apertura'] - yesterday['apertura']) / yesterday['apertura']    
        
        if yesterday['volumen'] == 0:
            vvolume[i] = 0
        else:    
            vvolume[i] = (row['volumen'] - yesterday['volumen']) / yesterday['volumen']
            
        if  yesterday['maximo'] == 0:
            vmax[i] = 0
        else:
            vmax[i] = (row['maximo'] - yesterday['maximo']) / yesterday['maximo']
            
        if yesterday['minimo'] == 0:
            vmin[i] = 0
        else:
            vmin[i] = (row['maximo'] - yesterday['minimo']) / yesterday['minimo']
        
        variab[i] = row['maximo']-row['minimo']
        
        if (yesterday['maximo']-yesterday['minimo']) == 0:
            vvariab[i] = 0
        else:
            vvariab[i] = ((row['maximo']-row['minimo']) - (yesterday['maximo']-yesterday['minimo'])) /  (yesterday['maximo']-yesterday['minimo'])

        
        if (reference_index != None) and (i > 1):
            good[i] = vcierre[i] - (reference_index['vcierre'])
        
        yesterday = row

    stock_values['var'] = variab
    stock_values['vvar'] = vvariab
    stock_values['vcierre'] = vcierre
    stock_values['vapertura'] = vapertura
    stock_values['vmax'] = vmax
    stock_values['vmin'] = vmin
    stock_values['vvolumen'] = vvolume
    
    if reference_index != None:
        stock_values['good'] = good
        
#################
##FILTERS       #
#################
def filter_data_frame(data, fil = "l"):
    '''
     Given a data frame, it filters high frequency noise using a given filter.
     
     Parameters-
     -l (default): for lowess
     -sv: savitzky_golay
     -b: Butterworth
     
    '''
    if fil == "sg":
        function_filter = savitzky_golay
    elif fil == "b":
        function_filter = butter_filter
    elif fil == "l":
        function_filter = low_filter
        
    for i in range(data.shape[1]):
        if isinstance(data.iloc[5,i], Number): #El 5 no va a ser nunca NA
            columna = np.array(data.iloc[:,i])
            data.iloc[:,i] = function_filter(columna)
    
    #return data
    
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

def _butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def _butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = _butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_filter(data, cutoff = 1500, fs = 50000):
    '''
    Applies the Butterworth filter to a set of data.
    '''
    return _butter_lowpass_filtfilt(data, cutoff, fs)

def low_filter(data, frac=0.025, it=0):
    '''
    Applies lowess filter to the data.
    '''
    return lowess(data, range(len(data)), is_sorted=True, frac=frac, it=it)[:,1]
            
