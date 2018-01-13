# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:44:19 2018

@author: Javier Fumanal Idocin
"""
import datetime
import pandas as pd
import numpy as np
from pandas_datareader import data
import quandl

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


def load_data_frame(path, sep='\t'):
    '''
    Loads finantial data from a text file separated by tabs.
    Returns a pandas data_frame.
    '''
    func = lambda dates: [datetime.datetime.strptime(x, '%d/%m/%Y') for x in dates]
    return pd.read_csv(path, sep=sep, parse_dates=['FECHA'],
                       date_parser=func,
                       dtype={"TICKER":np.str, "APERTURA":np.float, "MAXIMO":np.float,
                              "MINIMO":np.float, "ULTIMO":np.float, "VOLUMEN":np.float},
                       header=0,
                       names=["TICKER", "FECHA", "APERTURA", "MAXIMO", "MINIMO",
                              "ULTIMO", "VOLUMEN"],
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

    for i in np.arange(stock_values.shape[0]):
        row = stock_values.iloc[i]

        evolution[i] = row['ULTIMO'] - yesterday_stock
        vol_change[i] = row['VOLUMEN'] - yesterday_volume
        ascend[i] = row['ULTIMO'] > yesterday_stock
        variab[i] = row['MAXIMO']-row['MINIMO']
        variabp[i] = variab[i]/row['ULTIMO']

        yesterday_stock = row['ULTIMO']
        yesterday_volume = row['VOLUMEN']

        if reference_index != None:
            good[i] = row['ULTIMO'] - reference_index[row['FECHA']]['CIERRE']

    stock_values['VAR'] = variab
    stock_values['VARP'] = variabp
    stock_values['ASC'] = ascend
    stock_values['EVL'] = evolution
    stock_values['VCH'] = vol_change
    stock_values['IND'] = good

def load_stock_data(ticker, start=None, end=None, provider='quandl'):
    '''
    Given a ticker, it searches for it in quandl service.

    start-end: dates of the records.
    ¡Be sure it supports your schema!

    '''
    global AUTH_TOKEN

    if provider == 'quandl':
        df = quandl.get(ticker, authtoken=auth_token)
    else:
        df = data.DataReader(ticker, provider, start, end)

    df['APERTURA'] = df['Open']

    try:
        df['ULTIMO'] = df['Close']
    except KeyError:
        df['ULTIMO'] = df['Last']

    df['MINIMO'] = df['Low']
    df['MAXIMO'] = df['High']
    df['VOLUMEN'] = df['Volume']
    df['TICKER'] = ticker
    df['FECHA'] = df.index.values

    df = df.loc[:, ['APERTURA', 'ULTIMO', 'MINIMO', 'MAXIMO', 'VOLUMEN', 'TICKER', 'FECHA']]


    return df

def load_variables(path="./FD.json"):
    '''
    Load global variables: API token and tickers.

    '''
    global AUTH_TOKEN
    global TICKERS
    
    