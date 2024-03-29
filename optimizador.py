# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:39:29 2018

@author: Javier Fumanal Idocin

"""

import numpy as np
import progressbar
import heapq
import segmentation as sg
import clustering as cl
import pam 
from scipy.stats import norm

from random import randint
from sklearn.preprocessing import StandardScaler


#######################
# CONSTANTES
#######################
COTA_MINIMA = 3
#######################
#   VARIABLES GLOBALES (Mala practica, pero es por comodidad)
#######################
heap_vecinos = []
iterations = 0
nombres = []
light_segs = []
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
def normal_correction(mean, std, x):
    dist = norm(mean, std)
    
    return dist.pdf(x) / dist.pdf(mean)
def evaluate(datos, ponderaciones, silencios, carga_computacional=1,tipo_clustering="KMEANS", ncluster = 3, metrica = "SIL"):
    '''
    Dado un clasificador y un conjunto de train y test devuelve su tasa de acierto en test.
    '''
    global light_segs
    
    fecha = False
    if len(ponderaciones) == 0:
        return 0.0
    ligero = np.sign(carga_computacional) < 0
    carga_computacional = abs(carga_computacional)
    
    if carga_computacional < 1:
        datos = datos.sample(int(datos.shape[0] * carga_computacional))
        datos = datos.sort_index()
        carga_computacional = 1
        fecha = True
        
    if carga_computacional>=1:
        if not ligero:
            segmentos, _ = sg.segmentate_data_frame(df=datos, montecarlo=1, min_size=4, silence=silencios, 
                                                vector_importancias=ponderaciones, verbose=False)
            mean = len(sg.ultra_light_segmentation(datos, fecha=fecha))/2
        else:
            if len(light_segs) == 0:
                light_segs = sg.ultra_light_segmentation(datos, fecha=fecha) 
                
            segmentos = light_segs
            mean = len(segmentos)/2
            #mean = int(datos.shape[0]/100)
    
        if carga_computacional == 1:
            segmentados = cl.apply_segmentation(datos, segmentos, silencios, ponderaciones, fecha)
            
            if segmentados.shape[0] <= 6:
                return 0.0
            else:
                std = np.sqrt(mean)
                nsegs=[]
                for i in range(segmentados.shape[0]):
                    nsegs.append([i,i+1])
                segmentos = sg.join_segments(data=segmentados, o_segments=nsegs, distance=sg.interpretable_distance, threshold=0.5, minimum_size=1,silence=silencios, vector_importancias=ponderaciones)[0]
                segmentados = cl.apply_segmentation(segmentados, segmentos, silencios, ponderaciones, fecha)

                return cl.hopkins_statistic(cl.filter_numerical(segmentados),m=int(segmentados.shape[0]*0.5))* normal_correction(mean, std, len(segmentados))
            
        elif carga_computacional==2:
            if tipo_clustering == "DTW":
                segmentados = sg.get_segments(datos, segmentos)
                asignaciones = pam.kmedoids(segmentados, n_clus = ncluster)
                segments_df = cl.apply_clustering(datos, segmentos, asignaciones[1], asignaciones[2])
            elif tipo_clustering == "KMEANS":
                segments_df = cl.full_clustering(datos, segmentos, n_clus = ncluster, silencio=silencios, pesos = ponderaciones, normalizar=False)
                
            if metrica == "SIL":    
                return cl.sil_metric(segments_df[0])
            elif metrica == "ENTROPY":
                return entropy_metric(segments_df)
            

def discretizacion(final_df):
    '''
    Convierte valores continuos de variacion, volumen y cierre a valores discretos
    que marcan tendencia.
    
    Devuelve un numpy array con las columnas [cierre, volumen, var]
    '''     
    cierres = final_df['vcierre']
    volumen = final_df['vvolumen']
    var = final_df['var']
    
    cierres = ((np.sign(cierres)+1)/2).astype(int)
    var = np.abs(var) >  np.percentile(var, 90)
    vumbral = np.percentile(volumen, 75)
    subidas = (volumen > vumbral)*2
    bajadas = volumen < -vumbral
    volumen = volumen*0
    volumen = volumen.astype(int) + bajadas.astype(int) + subidas.astype(int) 
    
    return np.column_stack((cierres, volumen, var, final_df['cluster']))

def entropy_metric(X):
    '''
    Calculates the total entropy of  the clusterization made.
    '''
    res = 0
    dis_p = discretizacion(X)
    for i in np.unique(dis_p[:,-1]):
        dict_estados = {}
        cluster = dis_p[dis_p[:,-1]==i,:]
        for estado in range(cluster.shape[0]):
            prueba = str(cluster[estado,:])
            try:
                dict_estados[prueba] += 1
            except KeyError:
                dict_estados[prueba] = 1
            
        sum_estados = sum(dict_estados.values())
        for estado in dict_estados:
            res -= (dict_estados[estado]/sum_estados) * np.log(dict_estados[estado]/sum_estados)
      
    return res
        
    
def normalizar(data):
    '''
    Normalize data according to mean and muestral variance.
    '''
    dn = StandardScaler().fit_transform(X=data[:,:-1])
    return np.c_[dn, data[:,-1]]

###################################
#   EXPLORADOR - RECOLECTOR       #
###################################
    
def generar_nuevos(pesos, silencios, names):
    '''
    Dados unos pesos y unos grados genera tres variedades nuevas:
        -Un nuevo vector de pesos con un peso aumentado un 0.1.
        -Un nuevo vector de pesos con un peso disminuido un 0.1.
        -Silencia una variable.
        
    Devuelve - (Variante 1 de pesos, vairante 2 de pesos, variante de 1 grado)
    '''
    variable_probar = randint(0,len(pesos)/2-1)
    names = [x for x in names[0:int(len(names)/2)] if x not in silencios]
    
    nuevos_pesos1 = pesos.copy()
    nuevos_pesos2 = pesos.copy()
    nuevos_silencios = silencios.copy()
    pesos_silencio = pesos.copy()
    
    nuevos_pesos1[variable_probar] = nuevos_pesos1[variable_probar] - 0.1
    nuevos_pesos2[variable_probar] = nuevos_pesos2[variable_probar] + 0.1
    nuevos_silencios.append(names[variable_probar])
    
        #del pesos_silencio[variable_probar + int(len(pesos_silencio)/2)]
        #del pesos_silencio[variable_probar]
    pesos_silencio = np.delete(pesos_silencio, [variable_probar, variable_probar + int(len(pesos_silencio)/2)]  )
    
    return (nuevos_pesos1, nuevos_pesos2, [nuevos_silencios,pesos_silencio])

def evaluate_explorer(datos, ponderaciones, silencios, vecindario=False, carga=1):
    '''
    Deuvelve la tasa de acierto y con que clasificador se ha obtenido para
    una configuracion de pesos y grados.
    
    Si vecindario == True, evalua tambien el vecindario de la solucion
    
    Devuelve - (tasa de acierto, clasificador)
    '''
    if vecindario:
        resultado = evaluate(datos, ponderaciones, silencios, carga_computacional=carga)
        
        #Evaluate neighbourhood
        pesos1, pesos2, silencios2 = generar_nuevos(ponderaciones, silencios, list(cl.filter_numerical(datos)))
        
        sil1 = evaluate_explorer(datos, pesos1, silencios, False, carga)
        sil2 = evaluate_explorer(datos, pesos2, silencios,False, carga)
        sil3 = evaluate_explorer(datos, silencios2[1], silencios2[0], False, carga)
        
        vecindario_res = [sil1, sil2, sil3]
        
        return resultado, vecindario_res
    else:
        
        return evaluate(datos, ponderaciones, silencios, carga_computacional=carga)

##################################
#   COLMENA  (CONTROL)           #
##################################
def nueva_ronda(datos_originales, pesos, silencios, carga=1):
    '''
    Genera nuevos exploradores y devuelve la mejor tasa de acierto encontrada,
    con sus correspondientes pesos y grados.
    
    Tambien actualiza el conteo de clasificadores en caso de querer inhibir los que peor
    funcionen.
    
    Devuelve - (Tasa de acierto, pesos, grados, mejor)
    '''
    if len(silencios) > 0:
        pesos1, pesos2, silencios2 = generar_nuevos(pesos, silencios, list(cl.filter_numerical(datos_originales))) #Se generan los vecinos
    else:
        pesos1, pesos2, silencios2 = generar_nuevos(pesos, [], list(cl.filter_numerical(datos_originales))) #Se generan los vecinos
        
    #Se evaluan los exploradores
    sil1, vec1 = evaluate_explorer(datos_originales, pesos1, silencios, True, carga)
    sil2, vec2 = evaluate_explorer(datos_originales, pesos2, silencios, True, carga)
    
    if len(silencios2[1]) > 0:
        sil3, vec3 = evaluate_explorer(datos_originales, silencios2[1], silencios2[0], True, carga)
         
        #Nos quedamos con la mejor solucion encontrada y el mejor vecindario encontrado.
        mayor = np.argmax([sil1, sil2, sil3])
    else:
        mayor = np.argmax([sil1, sil2])
    if mayor == 2:
        solucion = [sil3, pesos, silencios2]
    elif mayor == 1:
        solucion = [sil2, pesos2, silencios]
    else:
        solucion = [sil1, pesos1, silencios]
    
    vec_sol1 = [vec1, pesos1, silencios]
    vec_sol2 = [vec2, pesos2, silencios]
    
    if len(silencios2[1]) > 0:
        vec_sol3 = [vec3, pesos, silencios2]
        vecs = [vec_sol1, vec_sol2, vec_sol3] 
    else:
        vecs = [vec_sol1, vec_sol2]
        
    solucion.append(vecs)
    
    return solucion
    

def simple_run(data, epochs = 30, reinicio = "random", silencios=["maximo","minimo","var"], carga=1):
    '''
    Realiza el proceso de optimizacion de unos datos etiquetados.
    
    epochs - numero de veces a ejecutar el proceso de busqueda.
    reinicio - politica de reinios si no se encuentra nada mejor en x epochs:
            -O bien se vuelve a un estado aleatorio.
            -O bien se vuelve al principio.
    '''
    global heap_vecinos,nombres
    
    #Se inician las variables de la solucion basica.
    rows, cols = cl.filter_numerical(data).shape
    nombres = list(cl.filter_numerical(data))
    pesos = [1] * (cols - (len(silencios)*2))
    iters_sin_mejora = 0
    
    #Se inicia la barra de progreso.
    bar = progressbar.ProgressBar()
    bar.max_value = epochs
    bar.min_value = 0
    bar.update(0)
    
    #Se calcula la solucion basica.
    mejor_tasa_de_acierto = evaluate_explorer(data, pesos, silencios, carga=carga)
    pesos_actuales = pesos
    grados_actuales = silencios
    grados = silencios
    
    #Se inicia el algoritmo
    for i in range(epochs):
        #Se evalua la solucion actual junto con sus vecindarios
        tasa_de_acierto, pesos_nuevos, grados_nuevos, vecindarios = nueva_ronda(data, pesos_actuales, grados_actuales,carga=carga)
        #Se anyaden los vecindarios al monticulo.
        add_vecindarios(vecindarios)
        #Si se mejora la solucion actual, se guarda
        if tasa_de_acierto > mejor_tasa_de_acierto:
            pesos = pesos_nuevos
            grados = grados_nuevos
            
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
            if len(grados_actuales) == 2 and isinstance(grados_actuales[0], list):
                pesos_actuales = grados_actuales[1]
                grados_actuales = grados_actuales[0]
        else:
            print("Se acaba por monticulo vacio")
            break
        #Se actualiza la barra de progreso.
        bar.update(i)
    
    #Se finaliza
    bar.finish()
    #Se vacia el heap (por si se quiere volver a ejecutar el algoritmo)
    heap_vecinos = []
    
    #Pequeno informe de como ha ido la ejecucion
    print("Reinicios %d" % (iters_sin_mejora / 5), "Epochs sin mejora: %d" % iters_sin_mejora, "(%d%%)" % int((100*(iters_sin_mejora*1.0) / epochs)))
    
    return mejor_tasa_de_acierto, pesos, grados
