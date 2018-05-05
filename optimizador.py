# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:39:29 2018

@author: Javier Fumanal Idocin

"""

import numpy as np
import progressbar
import heapq
from sklearn.preprocessing import StandardScaler
import segmentation as sg
import clustering as cl
import pam 

from random import randint

#######################
# CONSTANTES
#######################
COTA_MINIMA = 3
#######################
#   VARIABLES GLOBALES (Mala practica, pero es por comodidad)
#######################
heap_vecinos = []
iterations = 0
#########################

################################
#   GESTION DE VECINDARIOS     #
################################
def add_vecindarios(vecindarios, MAX_SIZE=10):
    '''
    Mete a los vecindarios en el heap en funcion de su calidad.
    '''
    global heap_vecinos
    
    for i in range(len(vecindarios)):
        tupla_insertar = [-evaluate_vecindario(vecindarios[i][0]), vecindarios[i][1:]] #Se multiplica por -1 porque el heap es de minimos
        heapq.heappush(heap_vecinos, tupla_insertar)
    
    #Nos aseguramos de que el heap no sea demasiado grande
    while len(heap_vecinos)>MAX_SIZE:
        del heap_vecinos[-1]
        
def get_best_vecindario():
    '''
    Devuelve una evaluacion de lo bueno que es el vecindario. En este caso, se
    usa la media de las tasas de acierto conseguidas con esos pesos y grados.
    '''
    return heapq.heappop(heap_vecinos)

def evaluate_vecindario(vec):
    '''
    Devuelve una puntuacion para el vecindario
    '''
    return np.mean(vec)

######################################
#   CLASIFICADORES - ENTRENAMIENTO   #
######################################
def evaluate(datos, ponderaciones, silencios, tipo_clustering="KMEANS", ncluster = 3):
    '''
    Dado un clasificador y un conjunto de train y test devuelve su tasa de acierto en test.
    '''
    #TODO
    segmentos, _ = sg.segmentate_data_frame(datos, min_size=7, silence=silencios, 
                                            vector_importancias=ponderaciones)
    
    if tipo_clustering == "DTW":
        segmentados = sg.get_segments(datos, segmentos)
        asignaciones = pam.kmedoids(segmentados, n_clus = ncluster)
        segments_df = cl.apply_clustering(datos, segmentos, asignaciones[1], asignaciones[2])
    elif tipo_clustering == "KMEANS":
        segments_df = cl.apply_segmentation(datos, segmentos, silencio = silencios, pesos = ponderaciones)
        segments_df = cl.full_clustering(segments_df, segmentos, n_clus = ncluster, silencio=silencios, pesos = ponderaciones)
        
        
    return cl.sil_metric(segments_df)

def normalizar(data):
    dn = StandardScaler().fit_transform(X=data[:,:-1])
    return np.c_[dn, data[:,-1]]

###################################
#   EXPLORADOR - RECOLECTOR       #
###################################
    
def generar_nuevos(pesos, silencios):
    '''
    Dados unos pesos y unos grados genera tres variedades nuevas:
        -Un nuevo vector de pesos con un peso aumentado un 0.1.
        -Un nuevo vector de pesos con un peso disminuido un 0.1.
        -Aumenta el grado de una variable.
        
    Devuelve - (Variante 1 de pesos, vairante 2 de pesos, variante de 1 grado)
    '''
    variable_probar = randint(0,len(pesos)-1)
    
    nuevos_pesos1 = pesos.copy()
    nuevos_pesos2 = pesos.copy()
    nuevos_silencios = silencios.copy()
    
    nuevos_pesos1[variable_probar] = nuevos_pesos1[variable_probar] - 0.1
    nuevos_pesos2[variable_probar] = nuevos_pesos2[variable_probar] + 0.1
    nuevos_silencios[variable_probar] = 0
    
    return [nuevos_pesos1, nuevos_pesos2, nuevos_silencios]

def evaluate_explorer(datos, ponderaciones, silencios, clasificadores_validos, vecindario=False):
    '''
    Deuvelve la tasa de acierto y con que clasificador se ha obtenido para
    una configuracion de pesos y grados.
    
    Si vecindario == True, evalua tambien el vecindario de la solucion
    
    Devuelve - (tasa de acierto, clasificador)
    '''
    
    if vecindario:
        resultado = evaluate(datos, ponderaciones, silencios)
        
        #Evaluate neighbourhood
        pesos1, pesos2, silencios2 = generar_nuevos(ponderaciones, silencios)
        
        sil1, _ = evaluate_explorer(datos, pesos1, silencios)
        sil2, _ = evaluate_explorer(datos, pesos2, silencios)
        sil3, _ = evaluate_explorer(datos, ponderaciones, silencios2)
        
        vecindario = [sil1, sil2, sil3]
        
        return resultado, vecindario
    else:
        
        return evaluate(datos, ponderaciones, silencios)

##################################
#   COLMENA  (CONTROL)           #
##################################
def nueva_ronda(datos_originales, pesos, silencios, clasificadores_validos):
    '''
    Genera nuevos exploradores y devuelve la mejor tasa de acierto encontrada,
    con sus correspondientes pesos y grados.
    
    Tambien actualiza el conteo de clasificadores en caso de querer inhibir los que peor
    funcionen.
    
    Devuelve - (Tasa de acierto, pesos, grados, mejor)
    '''
    pesos1, pesos2, silencios2 = generar_nuevos(pesos, grados) #Se generan los vecinos
    
    #Se evaluan los exploradores
    sil1, vec1 = evaluate_explorer(datos_originales, pesos1, grados, True)
    sil2, vec2 = evaluate_explorer(datos_originales, pesos2, grados, True)
    sil3, vec3 = evaluate_explorer(datos_originales, pesos, silencios2, True)
    
    
    #Nos quedamos con la mejor solucion encontrada y el mejor vecindario encontrado.
    mayor = np.argmax([tasa_acierto1, tasa_acierto2, tasa_acierto3])
    
    if mayor == 2:
        solucion = [tasa_acierto3, pesos, silencios2]
    elif mayor == 1:
        solucion = [tasa_acierto2, pesos2, silencios]
    else:
        solucion = [tasa_acierto1, pesos1, silencios]
    
    vec_sol1 = [vec1, pesos1, silencios]
    vec_sol2 = [vec2, pesos2, silencios]
    vec_sol3 = [vec3, pesos, silencios2]
    
    vecs = [vec_sol1, vec_sol2, vec_sol3] 
    
    solucion.append(vecs)
    
    return solucion
    

def simple_run(data, epochs = 30, reinicio = "random"):
    '''
    Realiza el proceso de optimizacion de unos datos etiquetados.
    
    epochs - numero de veces a ejecutar el proceso de busqueda.
    reinicio - politica de reinios si no se encuentra nada mejor en x epochs:
            -O bien se vuelve a un estado aleatorio.
            -O bien se vuelve al principio.
    '''
    global heap_vecinos
    
    #Se inician las variables de la solucion basica.
    rows, cols = data.shape
    pesos = [1] * (cols-1)
    silencios = [1] * (cols -1)
    iters_sin_mejora = 0
    
    #Se inicia la barra de progreso.
    bar = progressbar.ProgressBar()
    bar.max_value = epochs
    bar.min_value = 0
    bar.update(0)
    
    #Se calcula la solucion basica.
    mejor_tasa_de_acierto = evaluate_explorer(data, pesos, silencios)
    pesos_actuales = pesos
    grados_actuales = grados
    
    #Se inicia el algoritmo
    for i in range(epochs):
        #Se evalua la solucion actual junto con sus vecindarios
        tasa_de_acierto, pesos_nuevos, grados_nuevos, vecindarios = nueva_ronda(data, pesos_actuales, grados_actuales, clasificadores_validos)
        #Se anyaden los vecindarios al monticulo.
        add_vecindarios(vecindarios)
        #Si se mejora la solucion actual, se guarda
        if tasa_de_acierto > mejor_tasa_de_acierto:
            pesos = pesos_nuevos
            grados = grados_nuevos
            clasificador_mejor = clasificador_nuevo
            
            mejor_tasa_de_acierto = tasa_de_acierto
            iters_sin_mejora = 0
        else:
            #Si no, se anota.
            iters_sin_mejora += 1
            
            if  iters_sin_mejora == 10:
                heap_vecinos = []
                #Se reinicia de acuerdo a la politica de reinicio (solucion basica o aleatoria)
                if reinicio == "random":
                    pesos_actuales = np.random.randint(low=0, high=10, size=len(pesos))
                    pesos_actuales = pesos_actuales / 10.0
                elif reinicio == "ones":
                    pesos_actuales = [1] * len(pesos)
                    
                grados_actuales = [1] * (cols -1)
        
        #Si el monticulo no esta vacio, se coge el sujeto con mejor vecindario hasta ahora.
        if len(heap_vecinos)>0:
            vecino = get_best_vecindario()
            pesos_actuales = vecino[1][0]
            grados_actuales = vecino[1][1]
          
        #Se actualiza la barra de progreso.
        bar.update(i)
    
    #Se finaliza
    bar.finish()
    #Se vacia el heap (por si se quiere volver a ejecutar el algoritmo)
    heap_vecinos = []
    
    #Pequeno informe de como ha ido la ejecucion
    print("Reinicios %d" % (iters_sin_mejora / 5), "Epochs sin mejora: %d" % iters_sin_mejora, "(%d%%)" % int((100*(iters_sin_mejora*1.0) / epochs)))
    
    return mejor_tasa_de_acierto, pesos, grados
    
    
    
    