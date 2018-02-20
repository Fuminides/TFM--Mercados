# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:45:12 2018

@author: javi-
"""
import pandas as pd
import numpy as np

def _number_to_date(number_date):
    return str(number_date)[:10]

def _dates_to_string(colum_dates):
    res = []
    for i in range(len(colum_dates)):
        res.append(_number_to_date(colum_dates[i]))
    
    return res

def fill_dates(df):
    '''
    Fill the gaps in missing/non working days with linear interpolation.
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
            df['fecha'] = _dates_to_string(fecha)
            df['fecha'] = df['fecha'].apply(_number_to_date)
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

def _Bolinger_Bands(stock_price, window_size=15, num_of_std=3):

    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)

    return rolling_mean, upper_band, lower_band

def detect_finantial_outliers(df, variable, window_size=15):
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
            minos.append(i)
            
    return [minors, majors]

def crop_outliers(df, variable, window_size=20):
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
        
    
    