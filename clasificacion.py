# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:54:58 2018

@author: Javier Fumanal Idocin
"""
import numpy as np
import orangecontrib.associate.fpgrowth as fp

def extremos_incertidumbre(fit, datos, cluster):
    '''
    Devuelve la mayor y menor distancia de un conjunto de datos a un fit, dada una funcion
    de distancia.
    '''
    maximo = 0
    minimo = np.Inf
    
    for dato in datos:
        dist = distancia(fit, dato, cluster)
        
        if dist > maximo:
            maximo = dist
        elif dist < minimo:
            minimo = dist
    
    return [maximo, minimo]

def calcular_incertidumbre(fit, dato, confianzas, cluster):
    '''
    Calculate the confidence in a prediction.
    '''
    minimo = confianzas[1]
    maximo = confianzas[0]
    
    dist = distancia(fit, dato, cluster)
    
    confianza = (dist - minimo) / maximo
    
    if(confianza < 0):
        return 0
    else:
        return confianza
        
def distancia(fit, dato, cluster):
    '''
    Calcultes distances from points to cluters.
    '''
    return fit.transform(dato)[cluster]

def classify(datos, fit):
    '''
    Classifies data.
    '''
    return fit.predict(datos)

def tasa_aceptabilidad(incertidumbres):
    '''
    Gives the percentage of acceptable classifications made.
    '''
    incertidumbres2 = incertidumbres.copy()
    incertidumbres2[incertidumbres2 < 1] = 1.0 
    incertidumbres2[incertidumbres2 > 1] = 0.0
    
    return np.mean(incertidumbres2)

def transacciones_profundidad(X, profundidad = 1):
    '''
    '''
    clusters = np.array(X['cluster'])
    holder = np.zeros([len(clusters) - profundidad,profundidad+1])
    
    rows = holder.shape[0]
    cols = holder.shape[1]
    
    for i in range(rows):
        holder[i,cols-1] = clusters[i+profundidad]
        index = 0
        for j in range(i,i+profundidad):
            holder[i,index] = clusters[j]
    
    return holder

def rules_extractor(X, profundidades=range(4), metric = 0.4):
    res = {}
    
    for i in profundidades:
        T = transacciones_profundidad(X,i)
        
        itemsets = fp.frequent_itemsets(T, 2)
        itemsets = dict(fp.frequent_itemsets(T, metric))
        rules = [(P, Q, supp, conf) for P, Q, supp, conf in fp.association_rules(itemsets, metric)]
        
        res[i] = (itemsets, rules)
 
    return res       
        
        
    
    