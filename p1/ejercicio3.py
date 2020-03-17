import matplotlib.pyplot as plt
import numpy as np

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


"""Derivadas segundas y otras funciones necesarias para minimizar
f con el método de Newton"""

def f(x, y):
    return (x - 2) ** 2 \
           + 2 * ((y + 2) ** 2) \
           + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def dfx(x, y):
    return 2 * (x - 2) \
           + 4 * np.pi * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x)


def dfy(x, y):
    return 4 * (y + 2 + np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))

"""
Derivamos x otra vez
2-8*pi^2*sin(2*pi*y)*sin(2*pi*x)
"""
def dfx2(x, y):
    return 2 - 8 * (np.pi**2) * np.sin(2*np.pi*y)*np.sin(np.pi*2*x)

"""
Derivamos y otra vez
4-8*pi^2*sin(2*pi*x)*sin(2*pi*y)
"""
def dfy2(x, y):
    return 4 - 8 * (np.pi**2) * np.sin(2*np.pi*x) * np.sin(2* np.pi * y)
"""
Derivamos y despues de x
8*pi^2*cos(2*pi*x)*cos(2*pi*y)
(Hacerlo al revés es equivalente)
"""

def dfxy(x, y):
    return 8 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)

def H(x,y):
    return np.array([np.array([dfx2(x,y), dfxy(x,y)]), np.array([dfxy(x,y), dfy2(x,y)])])


def gradF(x, y):
    return np.array([dfx(x, y), dfy(x, y)])

def metodoDeNewton(initial_point, maxIter, eta):
    w = initial_point
    i = 0
    while i < maxIter:
        w -= eta * np.linalg.inv(H(w[0], w[1])).dot(gradF(w[0], w[1]))
        valor = f(w[0], w[1])
        valor_en_cada_iteration[i] = valor
        i += 1
    return w, i


etaF = 0.01
maxIterF = 50
initial_pointF = np.array([1.0, -1.0])
valor_en_cada_iteration = np.empty(maxIterF)
wF, itF = metodoDeNewton(initial_pointF, maxIterF, etaF)

print("Learning Rate: ", etaF)
imprimir_valor_por_iteration(valor_en_cada_iteration)

input("\n--- Pulsar tecla para continuar ---\n")
print('Coordenadas obtenidas: (', wF[0], ', ', wF[1], ')')
print('Mínimo obtenido: ', f(wF[0], wF[1]))
input("\n--- Pulsar tecla para continuar ---\n")

etaF = 0.1
initial_pointF = np.array([1.0, -1.0])
valor_en_cada_iteration = np.empty(maxIterF)

wF, itF = metodoDeNewton(initial_pointF, maxIterF, etaF)

print("Learning Rate: ", etaF)
imprimir_valor_por_iteration(valor_en_cada_iteration)
input("\n--- Pulsar tecla para continuar ---\n")
print('Coordenadas obtenidas: (', wF[0], ', ', wF[1], ')')
print('Mínimo obtenido: ', f(wF[0], wF[1]))
input("\n--- Pulsar tecla para continuar ---\n")


"""
Hacemos el mismo experimento que antes: 4 puntos y 2 learning rate
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
        wF, itF = metodoDeNewton(point, 50, etaF)
        minimos.append(f'{wF[0]}, {wF[1]}')
        valores.append(str(f(wF[0], wF[1])))

    header = ['Punto inicial', '\t\t\t Mínimo', 'Valor del mínimo']
    puntos = ['2.1, -2.1', '3, -3', '1.5, 1.5', '1, -1']
    makeTable(header, [puntos, minimos, valores])


compareInitialPoints(0.1)

input("\n--- Pulsar tecla para continuar ---\n")

compareInitialPoints(0.01)



