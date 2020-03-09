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


def printResults(f, w, it):
    # DISPLAY FIGURE
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


etaE = 0.1
maxIterE = 10000000000
error2getE = 1e-14
initial_pointE = np.array([1.0, 1.0])
wE, itE = gradient_descentE(initial_pointE, error2getE, maxIterE, etaE)

printResults(E, wE, itE)

input("\n--- Pulsar tecla para continuar ---\n")


# Seguir haciendo el ejercicio...

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
        i += 1
    return w, i


etaF = 0.01
maxIterF = 50
initial_pointF = np.array([1.0, -1.0])
wF, itF = gradient_descentF(initial_pointF, maxIterF, etaF)

print("Learning rate: ", etaF)
printResults(f, wF, itF)

input("\n--- Pulsar tecla para continuar ---\n")

etaF = 0.1

wF, itF = gradient_descentF(initial_pointF, maxIterF, etaF)

print("Learning rate: ", etaF)
printResults(f, wF, itF)

input("\n--- Pulsar tecla para continuar ---\n")


def initialPoint(x, y):
    return np.array([x, y])


def compareInitialPoints(etaF):
    initialPoints = [
        initialPoint(2.1, -2.1),
        initialPoint(3.0, -3.0),
        initialPoint(1.5, 1.5),
        initialPoint(1.0, -1.0),
    ]
    print("Para un learning rate de: ", etaF)
    for point in initialPoints:
        print("Punto inicial: ", point[0], ', ', point[1])
        wF, itF = gradient_descentF(point, maxIterF, etaF)
        print("Mínimo encontrado: ", wF[0], ', ', wF[1])
        print("Valor del mínimo: ", f(wF[0], wF[1]))


compareInitialPoints(0.1)

input("\n--- Pulsar tecla para continuar ---\n")

compareInitialPoints(0.01)



