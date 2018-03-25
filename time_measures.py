# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:21:32 2018

A little function to measure execution time by instrumentalization

@author: javi-
"""
import time

def exec_time(func, args, execuctions=10):
    '''
    Gives the execution time by instrumentalization of a given function.
    It executes it n times, and then it gives the average
    '''
    times = [0]*execuctions
    
    for i in range(execuctions):
        inicio = time.process_time()
        func(*args)
        final = time.process_time()
        
        times[i] = final - inicio
        
    return sum(times)/len(times)

def exec_ev(func, list_args, n=5):
    '''
    Gives the execution time by instrumentalization of a given function.
    It executes it for each set of args and gives  the result for all of them 
    according to exec_time() func.
    
    (In christian: use this to get exec time for different n size)
    '''
    times = [0]*len(list_args)
    
    for i in range(len(times)):
        times[i] = exec_time(func,list_args[i], n)
    
    return times
    