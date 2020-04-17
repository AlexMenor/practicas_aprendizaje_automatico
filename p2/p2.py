#!/usr/bin/env python
# coding: utf-8

# # Práctica 2 de AA
# ## Ejercicio 1

import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(4)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b



# Esta función separa los datos (matrix) en vectores, uno por cada columna
def getListForEachDimension(points):
  return points[:,0], points[:,1]
#Imprimimos la nube de puntos
def printCloudOfPoints(points, title):
  x, y = getListForEachDimension(points)
  fig, ax = plt.subplots()
  ax.scatter(x,y)
  ax.set_title(title)
  plt.show()



# EJERCICIO 1 a)

x = simula_unif(50, 2, [-50,50])

printCloudOfPoints(x, "Distribución de puntos uniforme")


input('Pulse cualquier tecla para continuar')

# EJERCICIO 1 b)

x = simula_gaus(50, 2, np.array([5,7]))

printCloudOfPoints(x, "Distribución de puntos Gaussiana")


input('Pulse cualquier tecla para continuar')



# Funciones necesarias para el apartado

def sign(x):
  if x >= 0:
    return 1
  else:
    return -1

def f2(x,y,a,b):
  valueUnsigned = y - a*x - b
  return sign(valueUnsigned)

# Lista de etiquetas dados los datos y los parámetros
# de la recta para poder clasificarlos

def getLabels(x,a,b):
  labels = []
  for i in range(len(x)):
    labels.append(f2(x[i][0], x[i][1],a,b))

  return labels



# Devuelve puntos generados para pintar la función en la gráfica

def getPointsFromFunction(f):
  puntosx = list(np.arange(-50,50,1))
  puntosy = []
  for x in puntosx:
    puntosy.append(f(x))

  
  return puntosx, puntosy

# Separa los puntos según las labels
# para que sea más fácil
# su impresión en matplotlib

def getPositivesAndNegatives(x,labels):
  positives = []
  negatives = []
  for xn, l in zip(x, labels):
    if l == 1:
      positives.append(xn)
    else:
      negatives.append(xn)

  return positives, negatives

# Función para imprimir los puntos con su clase y la recta que los clasifica

def printPointsWithClassificationF2(x, labels, f):
  fig, ax = plt.subplots()
  positives, negatives = getPositivesAndNegatives(x,labels)
  x,y = getListForEachDimension(np.array(positives))
  ax.scatter(x,y, c="blue", label="Positivos")
  x,y = getListForEachDimension(np.array(negatives))
  ax.scatter(x,y, c="orange", label="Negativos")
  ax.legend()
  x,y = getPointsFromFunction(f)
  ax.plot(x,y)
  ax.set_ylim(-50,50)
  ax.set_xlim(-50,50)
  plt.show()



## EJERCICIO 2 a)

np.random.seed(0)

# Generamos los puntos
x = simula_unif(50, 2, [-50,50])

#Generamos a y b
a,b = simula_recta([-50,50])

# Obtenemos las clases
labels = getLabels(x,a,b)

# Mostramos la gráfica
printPointsWithClassificationF2(x, labels, lambda x:a*x + b)

input('Pulse cualquier tecla para continuar')



# Para añadir estrictamente el ruido que se pide

def addNoiseToLabels(labels):
  numOfPositives = labels.count(1)
  positivesToDelete = numOfPositives * 0.1
  positivesToBecameNegatives = []

  numOfNegatives = labels.count(-1)
  negativesToDelete = numOfNegatives * 0.1
  negativesToBecamePositives = []

  while (positivesToDelete > 0):
    index = np.random.randint(0, len(labels)-1)
    if labels[index] == 1:
      positivesToBecameNegatives.append(index)
      positivesToDelete-=1

  while (negativesToDelete > 0):
    index = np.random.randint(0, len(labels)-1)
    if labels[index] == -1:
      negativesToBecamePositives.append(index)
      negativesToDelete-=1

    
  for i in positivesToBecameNegatives:
    labels[i] = -1
  
  for i in negativesToBecamePositives:
    labels[i] = 1



## EJERCICIO 2 b)
addNoiseToLabels(labels)

# Mostramos la gráfica
printPointsWithClassificationF2(x, labels, lambda x:a*x + b)
input('Pulse cualquier tecla para continuar')



# FUNCIÓN DADA EN LA PLANTILLA

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


    



# Lista de etiquetas dados los datos y la función para clasificarlos

def getLabelsFromAnyF(x,f):
  labels = []
  for i in range(len(x)):
    labels.append(sign(f(x[i][0], x[i][1])))

  return labels




labels = getLabelsFromAnyF(x, lambda x,y: (x-10)**2 + (y- 20)**2 - 400)
addNoiseToLabels(labels)
plot_datos_cuad(x,labels, lambda x : (x[:, 0] - 10) ** 2 + (x[:, 1] - 20) ** 2 - 400)

input('Pulse cualquier tecla para continuar')


labels = getLabelsFromAnyF(x, lambda x,y: 0.5*(x+10)**2 + (y- 20)**2 - 400)
addNoiseToLabels(labels)
plot_datos_cuad(x,labels, lambda x : 0.5 * (x[:, 0] + 10) ** 2 + (x[:, 1] - 20) ** 2 - 400)

input('Pulse cualquier tecla para continuar')


labels = getLabelsFromAnyF(x, lambda x,y: 0.5*(x-10)**2 - (y+ 20)**2 - 400)
addNoiseToLabels(labels)
plot_datos_cuad(x,labels, lambda x : 0.5 * (x[:, 0] - 10) ** 2 - (x[:, 1] + 20) ** 2 - 400)

input('Pulse cualquier tecla para continuar')



labels = getLabelsFromAnyF(x, lambda x,y: y - 20 * x **2 - 5*x + 3)
addNoiseToLabels(labels)
plot_datos_cuad(x,labels, lambda x : x[:, 1] - 20 * x[:,0]**2 - 5*x[:,0] + 3)


input('Pulse cualquier tecla para continuar')

# ## Ejercicio 2

#Añadimos un uno como primer elemento

def addOneAsFirstElement(x_original):
  x = np.zeros((len(x_original),3), dtype=float)

  for i in range(len(x_original)):
    x[i][0] = 1
    x[i][1] = x_original[i][0]
    x[i][2] = x_original[i][1]

  return x

#Función de error para problemas de clasificación

def error(data, labels, w):
  # x (dot) wt
  # n x 3 | 3 x 1 -> n x 1
  guessUnsigned = data.dot(w.reshape(3,-1))

  count = 0
  # Contamos los puntos mal clasificados
  for i in range(len(guessUnsigned)):
    if labels[i] != sign(guessUnsigned[i]):
      count += 1
  
  # Devolvemos el promedio

  return count / len(data)

# PLA

def pla(data, labels, max_iter, vini):
  # Copiamos la w inicial 
  w = vini.copy()

  hasConverged = False
  pasadas = 0

  # Si no hemos llegado al número máximo
  # de pasadas y no ha convergido...
  # SEGUIMOS!

  while pasadas != max_iter and not hasConverged:
    hasConverged = True
    for i in range(len(data)):
      guess = sign(w.dot(data[i].reshape(-1,1)))
      # En caso de fallar la predicción
      # ajustamos los pesos
      if guess != labels[i]:
        hasConverged = False
        w += labels[i] * data[i] 

    pasadas += 1

  return w, pasadas



# Función para pintar el ajuste 

def printFit(x, labels, w):
 positives, negatives = getPositivesAndNegatives(x,labels)
 fig, ax = plt.subplots()
 x,y = getListForEachDimension(np.array(positives))
 ax.scatter(x,y, c="blue", label="Positivos")
 x,y = getListForEachDimension(np.array(negatives))
 ax.scatter(x,y, c="orange", label="Negativos")
 
 
 puntosDeRectaX = np.arange(-50,50)
 puntosDeRectaY = []

 for i in range(len(puntosDeRectaX)):
   puntosDeRectaY.append((-w[0]-w[1]*puntosDeRectaX[i])/w[2])

 ax.plot(puntosDeRectaX, puntosDeRectaY, label="Ajuste")
 ax.legend()
 ax.set_xlim(-50,50)
 ax.set_ylim(-50,50)
 plt.show()

np.random.seed(44)

# Generamos puntos en [-50,50]

xOriginal = simula_unif(50, 2, [-50,50])

# Generamos una recta para clasificarlos

a,b = simula_recta([-50,50])

# Obtenemos sus etiquetas dados los parámetros que acabamos de generar

labels = getLabels(xOriginal,a,b)

x = addOneAsFirstElement(xOriginal)

# Ajustamos

w, pasadas = pla(x, labels, 500, np.zeros(3))

error_obtenido = error(x, labels, w)
print("PLA - Puntos linealmente separables - Con un vector de ceros")
print("Coordenadas",w)
print("Error", error_obtenido)
print("Iteraciones", pasadas)


printFit(xOriginal, labels, w)

input('Pulse cualquier tecla para continuar')


error_sum = 0
numOfIterations = []
for i in range(10):
  # De esta forma tenemos un vector en vez de una matriz de una fila
  # más cómodo para la función de printFit
  pesos_iniciales= np.random.uniform(0.0,1,(3,1)).reshape(-1)
  w, pasadas = pla(x, labels, 500, pesos_iniciales)
  error_obtenido = error(x, labels, w)
  error_sum += error_obtenido
  numOfIterations.append(pasadas)

print("PLA - Puntos linealmente separables - Puntos de arranque estocásticos")
print("Error medio", error_sum / 10)
print("Iteraciones de media", np.array(numOfIterations).mean())
print("Desviación típica de las iteraciones", np.array(numOfIterations).std())


input('Pulse cualquier tecla para continuar')

# Probamos ahora con el etiquetado "ruidoso"

addNoiseToLabels(labels)

w, pasadas = pla(x, labels, 500, np.zeros(3))

error_obtenido = error(x, labels, w)
print("PLA - Puntos NO linealmente separables - Con un vector de ceros")
print("Coordenadas",w)
print("Error", error_obtenido)
print("Iteraciones", pasadas)


printFit(xOriginal, labels, w)


input('Pulse cualquier tecla para continuar')
error_sum = 0
numOfIterations = []
for i in range(10):
  # De esta forma tenemos un vector en vez de una matriz de una fila
  # más cómodo para la función de printFit
  pesos_iniciales= np.random.uniform(0.0,1,(3,1)).reshape(-1)
  w, pasadas = pla(x, labels, 500, pesos_iniciales)
  error_obtenido = error(x, labels, w)
  error_sum += error_obtenido
  numOfIterations.append(pasadas)

print("PLA - Puntos NO linealmente separables - Arrancando estocásticamente")
print("Error medio", error_sum / 10)
print("Iteraciones de media", np.array(numOfIterations).mean())
print("Desviación típica de las iteraciones", np.array(numOfIterations).std())

input('Pulse cualquier tecla para continuar')


# Implemento el POCKET para ver si podemos mejorar en muestras ruidosas

def pocket(data, labels, max_iter, vini):
  # Copiamos la w inicial 
  w = vini.copy()

  hasConverged = False
  pasadas = 0

  best_w = w.copy()
  best_error = error(data, labels, w)

  # Si no hemos llegado al número máximo
  # de pasadas y no ha convergido...
  # SEGUIMOS!

  while pasadas != max_iter and not hasConverged:
    hasConverged = True
    for i in range(len(data)):
      guess = sign(w.dot(data[i].reshape(-1,1)))
      # En caso de fallar la predicción
      # ajustamos los pesos
      if guess != labels[i]:
        hasConverged = False
        w += labels[i] * data[i] 

    # Nos metemos al "pocket" la mejor
    # solución que hemos encontrado
    current_error = error(data, labels, w)
    
    if current_error < best_error:
      best_w = w.copy()
      best_error = current_error
    pasadas += 1

  return best_w, pasadas



w, pasadas = pocket(x, labels, 500, np.zeros(3))

error_obtenido = error(x, labels, w)
print("PLA-POCKET, datos NO linealmente separables. Arrancando con un vector de ceros")
print("Coordenadas",w)
print("Error", error_obtenido)
print("Iteraciones", pasadas)


printFit(xOriginal, labels, w)

input('Pulse cualquier tecla para continuar')

# Regresión logística
# Funciones de utilidad en este ejercicio

## Imprimimos los puntos obtenidos
def printPointsWithClassificationLogistic(x, labels, f):
  positives, negatives = getPositivesAndNegatives(x,labels)
  fig, ax = plt.subplots()
  x,y = getListForEachDimension(np.array(positives))
  ax.scatter(x,y, c="blue", label="Positivos")
  x,y = getListForEachDimension(np.array(negatives))
  ax.scatter(x,y, c="orange", label="Negativos")
  ax.legend()
  x,y = getPointsFromFunction(f)
  ax.plot(x,y)
  ax.set_xlim(0,2)
  ax.set_ylim(0,2)
  plt.show()

## Imprimimos los puntos con el ajuste
## obtenido
def printFitLogistic(x, labels, w):
  positives, negatives = getPositivesAndNegatives(x,labels)
  fig, ax = plt.subplots()
  x,y = getListForEachDimension(np.array(positives))
  ax.scatter(x,y, c="blue", label="Positivos")
  x,y = getListForEachDimension(np.array(negatives))
  ax.scatter(x,y, c="orange", label="Negativos")
  
  
  puntosDeRectaX = np.arange(0,20)
  puntosDeRectaY = []

  for i in range(len(puntosDeRectaX)):
    puntosDeRectaY.append((-w[0]-w[1]*puntosDeRectaX[i])/w[2])

  ax.plot(puntosDeRectaX, puntosDeRectaY, label="Ajuste")
  ax.legend()
  ax.set_xlim(0,2)
  ax.set_ylim(0,2)
  plt.show()




def sgd(x, y, eta, cambio_minimo, w=np.zeros(3)):

  cambio_actual = np.inf

  indices = np.arange(0, len(x))

  while cambio_actual >= cambio_minimo:
    
    # Nos quedamos con el w pre-iteración
    # para poder hacer el incremento después
    w_anterior = w.copy()

    # Permutación 
    np.random.shuffle(indices)
    
    for i in indices:
      xn = x[i]
      yn = y[i]
      gt = -( yn * xn ) / ( 1 + np.exp(yn * np.dot(w,xn))) 
      w -= eta * gt

    cambio_actual = np.linalg.norm(w_anterior - w)

  return w




def errorSgd(x,y,w):
  error = 0
  for xn, yn in zip(x, y):
    error += np.log(1 + np.exp(-yn * w.dot(xn)))

  return error / len(x)
        



## Recta que corta el cuadro [0,2] X [0,2]
a,b = simula_recta([0,2])

## Puntos en ese cuadro, probabilidad uniforme
xOriginal = simula_unif(100,2,[0,2])

# Obtenemos las clases
y = getLabels(xOriginal,a,b)

printPointsWithClassificationLogistic(xOriginal, y, lambda x: a*x+b)
input('Pulse cualquier tecla para continuar')

# Añadimos el 1 al principio

x = addOneAsFirstElement(xOriginal)



# Ajustamos
w = sgd(x,y,0.01, 0.01)

printFitLogistic(xOriginal, y,w)
print("Ajuste de regresión logística con datos linealmente separables")
print("Error Ein", errorSgd(x,y,w))

input('Pulse cualquier tecla para continuar')


xOriginalNuevasMuestras = simula_unif(1000,2,[0,2])


# Obtenemos las clases
yNuevasMuestras = getLabels(xOriginalNuevasMuestras, a,b)

printPointsWithClassificationLogistic(xOriginalNuevasMuestras, yNuevasMuestras, lambda x: a*x+b)
input('Pulse cualquier tecla para continuar')

# Añadimos el 1 al principio

xNuevasMuestras = addOneAsFirstElement(xOriginalNuevasMuestras)




# Imprimimos los nuevos puntos con el ajuste que
# obtuvimos antes
printFitLogistic(xOriginalNuevasMuestras, yNuevasMuestras,w)
print("Ajuste de regresión logística con datos linealmente separables")
print("Error Eout", errorSgd(xNuevasMuestras,yNuevasMuestras,w))

input('Pulse cualquier tecla para continuar')

# ## EJERCICIO 3


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

def printEjer3(x,y, w = [], training = True):
  if training:
    title = "Dígitos manuscritos (TRAINING)"
  else:
    title = "Dígitos manuscritos (TEST)"
  #mostramos los datos
  fig, ax = plt.subplots()
  ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
  ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
  ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title=title)
  ax.set_xlim((0, 1))

  if len(w) != 0:

    puntosDeRectaX = np.arange(0,1,0.01)
    puntosDeRectaY = []

    for i in range(len(puntosDeRectaX)):
      puntosDeRectaY.append((-w[0]-w[1]*puntosDeRectaX[i])/w[2])

    ax.plot(puntosDeRectaX, puntosDeRectaY, label="Ajuste")
    ax.legend()
    ax.set_ylim(-7,-1)

  plt.legend()
  plt.show()



# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

printEjer3(x,y, training=True)
input('Pulse cualquier tecla para continuar')
printEjer3(x_test,y_test,training=False)

input('Pulse cualquier tecla para continuar')



# Entreno con algoritmo de regresión logística
w = sgd(x,y,0.01,0.01)

print("Aplicando SGD logistic regression")
# Ein
print("Error Ein de regresión logística", errorSgd(x,y,w))
ein_sgd_solo = error(x,y,w)
print("Error Ein de clasificación", ein_sgd_solo)

printEjer3(x,y,w,True)
input('Pulse cualquier tecla para continuar')
# Eout
print("Error Eout de regresión logística", errorSgd(x_test,y_test,w))
etest_sgd_solo = error(x_test,y_test,w)
print("Error Eout de clasificación", etest_sgd_solo)
printEjer3(x_test,y_test,w,False)

input('Pulse cualquier tecla para continuar')



# Aprovecho el w que he obtenido en el paso anterior y lo utilizo como pesos iniciales del PLA-Pocket.
w, pasadas = pocket(x,y,1000,w)

print("Aplicando PLA-Pocket después de logistic regression")
print("Error Ein ", error(x,y,w))
printEjer3(x,y,w,True)
input('Pulse cualquier tecla para continuar')


print("Error Eout ", error(x_test,y_test,w))
printEjer3(x_test,y_test,w,False)
input('Pulse cualquier tecla para continuar')




w, pasadas = pocket(x,y,1000,np.zeros(3))
ein_pla_solo = error(x,y,w)
print("Aplicando solo PLA-POCKET, arrancando con ceros")
print("Error Ein ", ein_pla_solo)
printEjer3(x,y,w,True)
input('Pulse cualquier tecla para continuar')
etest_pla_solo = error(x_test,y_test,w)
print("Error Eout ", etest_pla_solo)
printEjer3(x_test,y_test,w,False)
input('Pulse cualquier tecla para continuar')



w = sgd(x,y,0.01,0.01,w)

print("Aplicando SGD logistic regression al w obtenido con PLA-Pocket")
# Ein
print("Error Ein de regresión logística", errorSgd(x,y,w))
ein_pla_sgd = error(x,y,w)
print("Error Ein de clasificación", ein_pla_sgd)
printEjer3(x,y,w,True)
input('Pulse cualquier tecla para continuar')
# Eout
print("Error Eout de regresión logística", errorSgd(x_test,y_test,w))
etest_pla_sgd = error(x_test,y_test,w)
print("Error Eout de clasificación", error(x_test,y_test,w))
printEjer3(x_test,y_test,w,False)
input('Pulse cualquier tecla para continuar')



N = len(x)
N_test = len(x_test)

print("Cotas para EOUT")

cota_eout_sgd_solo = ein_sgd_solo + np.sqrt(np.log(2/0.05)/(2*N))
cota_etest_sgd_solo = etest_sgd_solo + np.sqrt(np.log(2/0.05)/(2*N_test))
print("Eout para sgd solo (utilizando ein) es: ", cota_eout_sgd_solo)
print("Eout para sgd solo (utilizando etest) es: ", cota_etest_sgd_solo)

print("\n")

cota_eout_pla_solo = ein_pla_solo + np.sqrt(np.log(2/0.05)/(2*N))
cota_etest_pla_solo = etest_pla_solo + np.sqrt(np.log(2/0.05)/(2*N_test))
print("Eout para PLA solo (utilizando ein) es: ", cota_eout_pla_solo)
print("Eout para PLA solo (utilizando etest) es: ", cota_etest_pla_solo)

print("\n")

cota_eout_pla_sgd = ein_pla_sgd + np.sqrt(np.log(2/0.05)/(2*N))
cota_etest_pla_sgd = etest_pla_sgd + np.sqrt(np.log(2/0.05)/(2*N_test))
print("Eout para PLA-SGD (utilizando ein) es: ", cota_eout_pla_sgd)
print("Eout para PLA-SGD (utilizando etest) es: ", cota_etest_pla_sgd)

