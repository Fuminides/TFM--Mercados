# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:44:19 2018

@author: Javier Fumanal Idocin
"""
import datetime
import pandas as pd
import numpy as np

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
