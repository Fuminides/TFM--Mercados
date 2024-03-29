# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:44:19 2018

@author: Javier Fumanal Idocin
"""
import json
import pandas as pd
import numpy as np
from pandas_datareader import data
from datetime import datetime
from pathlib import Path
import os.path
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

def load_schema(file="./Datos/Schemas.json"):
    '''
    Loads file that contains a Database schema and maps it to known fields.
    '''
    return json.load(open(file))

def select_schema(name):
    '''
    Given a name from Quandl, it returns
    '''
    schemas = load_schema()
    try:
        return schemas[name.split("\\")[0]]
    except KeyError:
        return None

def load_local_data(path, schema = None, sep='\t', dec=","):
    '''
    Loads finantial data from a text file separated by tabs.
    Returns a pandas data_frame.
    '''
    
    if schema is None:
        func = lambda dates: [datetime.strptime(x, '%d/%m/%Y') for x in dates]
        df_res = pd.read_csv(path, sep=sep, parse_dates=['fecha'],
                       date_parser=func,
                       dtype={"ticker":np.str, "apertura":np.float, "maximo":np.float,
                              "minimo":np.float, "cierre":np.float, "volumen":np.float},
                       header=0,
                       names=["ticker", "fecha", "apertura", "maximo", "minimo",
                              "cierre", "volumen"],
                       decimal=",")
        df_res.index = df_res['fecha']
        
        return df_res
    else:
        func = lambda dates: [datetime.strptime(x, '%Y-%m-%d') for x in dates]
        carga_inicial = pd.read_csv(path, sep = sep)
        nombres = list(carga_inicial)
        nombres_ordenados = []
        tipos = {}
        dumpings = 0
        for nombre in nombres:
            try:
                nombres_ordenados.append(schema[nombre])
                if schema[nombre] != "fecha":
                    tipos[schema[nombre]] = np.float

            except KeyError:
                nombres_ordenados.append('dump' + str(dumpings))
                dumpings += 1

        df_res =  pd.read_csv(path, sep=sep, parse_dates=['fecha'],
                       date_parser=func,
                       dtype=tipos,
                       header=0,
                       names=nombres_ordenados,
                       skiprows = 0,
                       decimal=dec)
        
        for i in np.arange(0,dumpings):
            df_res = df_res.drop('dump' + str(i), 1)
        
        df_res.index = df_res['fecha']
        
        return df_res
        


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

    if schema is None:
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

def init_variables(path="./Datos/FD.json"):
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
        
        if key[2] == "online":
            dfs[ticker] = load_online_data(key)
        elif key[2] == "offline":
            schem = select_schema(ticker)
            dfs[ticker] = load_local_data(key[3], schem, dec=".")
            dfs[ticker]['ticker'] =  ticker.upper()

    return dfs

def _dates_to_string(colum_dates):
    res = []
    for i in range(len(colum_dates)):
        res.append(_number_to_date(colum_dates[i]))

    return res

def _number_to_date(number_date):
    return str(number_date)[:10]

def save_df(df):
    '''
    '''
    df.to_pickle('./Datos/' + df.ticker[0].split("/")[-1] + '.pkl')

def restore_df(ticker):
    '''
    '''
    df_file = Path('./Datos/' + ticker.split("/")[-1] + '.pkl')
    if os.path.isfile(df_file):
        return pd.read_pickle(df_file)
    else:
        return None
    