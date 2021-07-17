# Librerías útiles para el módulo Machine Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from scipy import stats


# Subconjuntos de entrenamiento, validación y prueba
def particionar(entradas, salidas, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    temp_size = porcentaje_validacion + porcentaje_prueba
    print(temp_size)
    x_train, x_temp, y_train, y_temp = train_test_split(entradas, salidas, test_size =temp_size)
    if(porcentaje_validacion > 0):
        test_size = porcentaje_prueba/temp_size
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = test_size)
    else:
        return [x_train, None, x_temp, y_train, None, y_temp]
    return [x_train, x_val, x_test, y_train, y_val, y_test]

# Métricas básicas de clasificación binaria
def metricasclass(matrizconfusion):
    """
    Calcula las métricas básicas de un modelo de clasificación binario.

    Parameters
    ----------
    matrizconfusion : 2D-Array
        Matriz de confusión con las siguientes entradas
        (0,0) = TN, (0,1) = FP, (1,0) = FN y (1,1) = TP
        
    Returns
    -------
    precision : Float
        Porcentaje de observaciones correctamente clasificadas.
    sensibilidad : Float
        Porcentaje de observaciones positivas correctamente clasificadas.
    especificidad : Float
        Porcentaje de observaciones negativas correctamente clasificadas.

    """
    (TN, FP, FN, TP) = matrizconfusion.ravel()
    precision = ((TP + TN) / (TP + TN + FP + FN)) * 100
    sensibilidad = (TP / (TP + FN)) * 100
    especificidad = (TN / (TN + FP)) * 100
    
    return precision, sensibilidad, especificidad



