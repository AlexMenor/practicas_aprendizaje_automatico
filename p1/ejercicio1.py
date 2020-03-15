# -*- coding: utf-8 -*-
"""
TRABAJO 1.
Nombre Estudiante: Alejandro Menor Molinero 13174410X
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')


""" 
Como hay que imprimir la gráfica/resultados de ambas funciones
extraigo ese bloque de código para reutilizarlo
"""
def printResults(f, w, it):
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-30, 30, 50)
    y = np.linspace(-30, 30, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                           cstride=1, cmap='jet')
    min_point = np.array([w[0], w[1]])
    min_point_ = min_point[:, np.newaxis]
    ax.plot(min_point_[0], min_point_[1], f(min_point_[0], min_point_[1]), 'r*', markersize=10)
    ax.set(title='Función sobre la que se calcula el descenso de gradiente')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('f(u,v)')
    fig.show()

    print('Numero de iteraciones: ', it)
    print('Coordenadas obtenidas: (', w[0], ', ', w[1], ')')
    print('Mínimo obtenido: ', f(w[0], w[1]))


def E(u, v):
    return (u * (np.e ** v) - 2 * v * (np.e ** (-u))) ** 2


# Derivada parcial de E con respecto a u
def dEu(u, v):
    return 2 * (u * (np.e ** v) - 2 * v * (np.e ** (-u))) * ((np.e ** v) + 2 * v * (np.e ** (-u)))


# Derivada parcial de E con respecto a v
def dEv(u, v):
    return 2 * (u * (np.e ** v) - 2 * v * (np.e ** (-u))) * (u * (np.e ** v) - 2 * (np.e ** (-u)))


# Gradiente de E
def gradE(u, v):
    return np.array([dEu(u, v), dEv(u, v)])


def gradient_descentE(initial_point, error2get, maxIter, eta):
    w = initial_point
    i = 0
    while i < maxIter and E(w[0], w[1]) > error2get:
        w -= eta * gradE(w[0], w[1])
        i += 1
    return w, i

"""
Considerar la función E(u, v) = (uev − 2ve−u )2 . Usar gradiente descendente
para encontrar un mínimo de esta función, comenzando desde el punto (u, v) = (1, 1) y
usando una tasa de aprendizaje η = 0,1.

"""
etaE = 0.1
maxIterE = 10000000000
error2getE = 1e-14
initial_pointE = np.array([1.0, 1.0])
wE, itE = gradient_descentE(initial_pointE, error2getE, maxIterE, etaE)

print("Función E")

printResults(E, wE, itE)

input("\n--- Pulsar tecla para continuar ---\n")

"""
Función para imprimir el valor de la función a minimizar
en cada iteración
"""
def imprimir_valor_por_iteration(valores_por_iteracion):
    fig, ax = plt.subplots()
    ax.plot(valores_por_iteracion)

    ax.set(xlabel='Iteración', ylabel='f(w)',
           title='Valor de la función a optimizar en cada iteración')
    ax.grid()

    plt.show()

def f(x, y):
    return (x - 2) ** 2 \
           + 2 * ((y + 2) ** 2) \
           + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def dfx(x, y):
    return 2 * (x - 2) \
           + 4 * np.pi * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x)


def dfy(x, y):
    return 4 * (y + 2 + np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))


def gradF(x, y):
    return np.array([dfx(x, y), dfy(x, y)])


def gradient_descentF(initial_point, maxIter, eta):
    w = initial_point
    i = 0
    while i < maxIter:
        w -= eta * gradF(w[0], w[1])
        valor = f(w[0], w[1])
        valor_en_cada_iteration[i] = valor
        i += 1
    return w, i


"""
(2 puntos) Considerar ahora la función f (x, y) = (x − 2)2 + 2(y + 2)2 + 2 sin(2πx) sin(2πy)
a) Usar gradiente descendente para minimizar esta función. Usar como punto inicial
(x0 = 1, y0 = −1), (tasa de aprendizaje η = 0,01 y un máximo de 50 iteraciones.
Generar un gráfico de cómo desciende el valor de la función con las iteraciones. Repetir
"""
etaF = 0.01
maxIterF = 50
initial_pointF = np.array([1.0, -1.0])
valor_en_cada_iteration = np.empty(maxIterF)
wF, itF = gradient_descentF(initial_pointF, maxIterF, etaF)

print("Función F")
print("Learning rate: ", etaF)
printResults(f, wF, itF)

input("\n--- Pulsar tecla para continuar ---\n")
print("Learning Rate: ", etaF)
imprimir_valor_por_iteration(valor_en_cada_iteration)
input("\n--- Pulsar tecla para continuar ---\n")

etaF = 0.1
initial_pointF = np.array([1.0, -1.0])
valor_en_cada_iteration = np.empty(maxIterF)

wF, itF = gradient_descentF(initial_pointF, maxIterF, etaF)

print("Función F")
print("Learning rate: ", etaF)
printResults(f, wF, itF)
input("\n--- Pulsar tecla para continuar ---\n")
print("Learning Rate: ", etaF)
imprimir_valor_por_iteration(valor_en_cada_iteration)
input("\n--- Pulsar tecla para continuar ---\n")

"""
Obtener el valor mínimo y los valores de las variables (x, y) en donde se alcanzan
cuando el punto de inicio se fija en: (2,1, −2,1), (3, −3),(1,5, 1,5),(1, −1). Generar una
tabla con los valores obtenidos
"""

"""
Función para generar una tabla, fuente: https://stackoverflow.com/questions/51730186/how-to-generate-table-using-python
"""
def makeTable(headerRow,columnizedData,columnSpacing=2):
    """Creates a technical paper style, left justified table

    Author: Christopher Collett
    Date: 6/1/2019"""
    from numpy import array,max,vectorize

    cols = array(columnizedData,dtype=str)
    colSizes = [max(vectorize(len)(col)) for col in cols]

    header = ''
    rows = ['' for i in cols[0]]

    for i in range(0,len(headerRow)):
        if len(headerRow[i]) > colSizes[i]: colSizes[i]=len(headerRow[i])
        headerRow[i]+=' '*(colSizes[i]-len(headerRow[i]))
        header+=headerRow[i]
        if not i == len(headerRow)-1: header+=' '*columnSpacing

        for j in range(0,len(cols[i])):
            if len(cols[i][j]) < colSizes[i]:
                cols[i][j]+=' '*(colSizes[i]-len(cols[i][j])+columnSpacing)
            rows[j]+=cols[i][j]
            if not i == len(headerRow)-1: rows[j]+=' '*columnSpacing

    line = '-'*len(header)
    print(line)
    print(header)
    print(line)
    for row in rows: print(row)
    print(line)

def initialPoint(x, y):
    return np.array([x, y])


def compareInitialPoints(etaF):
    initialPoints = [
        initialPoint(2.1, -2.1),
        initialPoint(3.0, -3.0),
        initialPoint(1.5, 1.5),
        initialPoint(1.0, -1.0),
    ]
    minimos = []
    valores = []
    print("Para un learning rate de: ", etaF)
    for point in initialPoints:
        wF, itF = gradient_descentF(point, maxIterF, etaF)
        minimos.append(f'{wF[0]}, {wF[1]}')
        valores.append(str(f(wF[0], wF[1])))

    header = ['Punto inicial', '\t\t\t Mínimo', 'Valor del mínimo']
    puntos = ['2.1, -2.1', '3, -3', '1.5, 1.5', '1, -1']
    makeTable(header, [puntos, minimos, valores])


compareInitialPoints(0.1)

input("\n--- Pulsar tecla para continuar ---\n")

compareInitialPoints(0.01)



