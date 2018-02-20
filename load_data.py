# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:44:19 2018

@author: Javier Fumanal Idocin
"""
import json
import quandl


import pandas as pd
import numpy as np
from pandas_datareader import data
from datetime import datetime

##############################
# VARIABLES                  #
##############################
#Authentification token if you have one for online data provider
AUTH_TOKEN = None
#Tickers that you are interested in.
TICKERS = []

##############################
# FUNCTIONS                  #
##############################

def load_schema(file = "./Schemas.json"):
    '''
    Loads file that contains a Database schema and maps it to known fields.
    '''
    return json.load(open(file))

def select_schema(name):
    '''
    Given a name from Quandl, it returns
    '''
    schemas = load_schema()
    return schemas[name.split("\\")[0]]

def load_local_data(path, sep='\t'):
    '''s
    Loads finantial data from a text file separated by tabs.
    Returns a pandas data_frame.
    '''
    func = lambda dates: [datetime.datetime.strptime(x, '%d/%m/%Y') for x in dates]
    return pd.read_csv(path, sep=sep, parse_dates=['fecha'],
                       date_parser=func,
                       dtype={"ticker":np.str, "apertura":np.float, "maximo":np.float,
                              "minimo":np.float, "cierre":np.float, "volumen":np.float},
                       header=0,
                       names=["ticker", "fecha", "apertura", "maximo", "minimo",
                              "cierre", "volumen"],
                       decimal=",")

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
    variabp = np.zeros(stock_values.shape[0])
    ascend = np.zeros(stock_values.shape[0])
    evolution = np.zeros(stock_values.shape[0])

    if reference_index != None:
        good = np.zeros(stock_values.shape[0])

    vol_change = np.zeros(stock_values.shape[0])

    yesterday_stock = 0
    yesterday_volume = 0
    yesterday_date = '03/01/1930'
    
    for i in np.arange(stock_values.shape[0]):
        row = stock_values.iloc[i]

        evolution[i] = row['cierre'] - yesterday_stock
        vol_change[i] = row['volumen'] - yesterday_volume
        ascend[i] = row['cierre'] > yesterday_stock
        variab[i] = row['maximo']-row['minimo']
        variabp[i] = variab[i]/row['cierre']

        yesterday_stock = row['cierre']
        yesterday_volume = row['volumen']
        
        if (reference_index != None) and (i > 1):
            good[i] = evolution[i] - (reference_index[row['fecha']]['cierre']-reference_index[yesterday_date]['cierre'])
        
        yesterday_date = row['fecha']

    stock_values['var'] = variab
    stock_values['varp'] = variabp
    stock_values['asc'] = ascend
    stock_values['evl'] = evolution
    stock_values['vch'] = vol_change
    stock_values['ind'] = good

def load_online_data(ticker, schema=None, start=None, end=None, provider='quandl'):
    '''
    Given a ticker, it searches for it in quandl service.

    start-end: dates of the records.
    ¡Be sure it supports your schema!

    '''
    global AUTH_TOKEN

    if provider == 'quandl':
        df = quandl.get(ticker[0], authtoken=AUTH_TOKEN)
    else:
        df = data.DataReader(ticker[0], provider, start, end)
    
    if schema==None:
        if ticker[1] == "index":
            df['cierre'] = df['Value']
            df = df.loc[:, ['cierre']]
        else:
            df['apertura'] = df['Open']
    
            try:
                df['cierre'] = df['Close']
            except KeyError:
                df['cierre'] = df['Last']
    
            df['minimo'] = df['Low']
            df['maximo'] = df['High']
            df['volumen'] = df['Volume']
            df['ticker'] = ticker[0]
            df['fecha'] = _dates_to_string(df.index.values)
            df['fecha'] = df['fecha'].apply(_number_to_date)
    else:
        df['cierre'] = df[schema['cierre']]
        df['apertura'] = df[schema['apertura']]
        df['minimo'] = df[schema['minimo']]
        df['maximo'] = df[schema['maximo']]
        df['volumen'] = df[schema['volumen']]
        df['fecha'] = df[schema['fecha']]
        df['ticker'] = df[schema['ticker']]
            

    df = df.loc[:, ['apertura', 'cierre', 'minimo', 'maximo', 'volumen', 'ticker', 'fecha']]


    return df

def init_variables(path="./FD.json"):
    '''
    Init global variables: API token and tickers.

    '''
    global AUTH_TOKEN
    global TICKERS

    data_json = json.load(open(path))

    AUTH_TOKEN = data_json['token']
    TICKERS = data_json['ticker']

def load_full_stock_data(reload=True):
    '''
    Loads into memory all the tickers listed in the DF.json file.
    '''
    if reload:
        init_variables()

    dfs = {}

    for ticker in TICKERS:
        key = TICKERS[ticker]
        dfs[ticker] = load_online_data(key)

    return dfs

def filter_numerical(df):
    '''
    Returns a data frame only with the financial numerical values.
    (apertura, cierre, minimo, maximo, volumen)
    '''
    return df.loc[:, df.columns - ["ticker"] - ["fecha"]]

def _dates_to_string(colum_dates):
    res = []
    for i in range(len(colum_dates)):
        res.append(_number_to_date(colum_dates[i]))
    
    return res
        
def _number_to_date(number_date):
    return str(number_date)[:10]