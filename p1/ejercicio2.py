# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Alejandro Menor Molinero 13174410X
"""
import math

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
"""
def Err(x,y,w):
	error = 0
	for i in range(len(x)):
		guess = w.dot(x[i])

		error += ((guess - y[i]) ** 2)

	return error / len(x)
"""
# Optimizada con operaciones matriciales
# Reshape (-1,1) pone el vector como una sola columna
def Err(x,y,w):
	errorBruto = x.dot(w.reshape(-1, 1)) - y.reshape(-1, 1)
	errorCuadratico = np.square(errorBruto)

	return errorCuadratico.mean()

# Gradiente Descendente

def gd (x, y, eta, maxiter):
	w = np.array([0, 0, 0], dtype=float)
	for i in range(maxiter):
		for j in range(len(w)):
			sum = 0.0
			for n in range(len(x)):
				guess = w.dot(x[n])
				sum += (x[n][j] * (guess - y[n]))
			w[j] -= (eta * (2 *sum)/len(x))

	return w

# Gradiente Descendente Estocastico
"""
def sgd(x, y, eta, number_of_minibatches, maxiter):
	mini_batches_x = np.array_split(x, number_of_minibatches)
	mini_batches_y = np.array_split(y, number_of_minibatches)
	w = np.array([0, 0, 0], dtype=float)

	index_array = np.arange(0, number_of_minibatches)

	for i in range(maxiter):
		np.random.shuffle(index_array)
		for random_index in index_array:
			xs = mini_batches_x[random_index]
			ys = mini_batches_y[random_index]
			for j in range(len(w)):
				sum = 0
				for n in range(len(xs)):
					guess = w.dot(xs[n])
					sum += (xs[n][j] * (guess - ys[n]))
				w[j] -= (eta * (2*sum)/len(xs))

	return w
"""
# Optimizada con operaciones matriciales
def sgd(x, y, eta, number_of_minibatches, maxiter):
	mini_batches_x = np.array_split(x, number_of_minibatches)
	mini_batches_y = np.array_split(y, number_of_minibatches)
	w = np.array([0, 0, 0], dtype=float)

	index_array = np.arange(0, number_of_minibatches)

	for i in range(maxiter):
		np.random.shuffle(index_array)
		for random_index in index_array:
			xs = mini_batches_x[random_index]
			ys = mini_batches_y[random_index]
			error_actual = xs.dot(w) - ys
			"""(xs[n][j] * (guess - ys[n])) pero matricial"""
			error_por_x = xs * error_actual.reshape(-1, 1)
			""" Hacemos la media de las columnas nos queda vector 1xd"""
			correcion = np.mean(error_por_x, axis=0)
			""" Corregimos w"""
			w -= (2 * eta * correcion)

	return w



# Pseudoinversa
def pseudoinverse(x,y):
	pseudoinverse = np.linalg.pinv(x)
	return pseudoinverse.dot(y)


"""Como era recurrente pintar el error
y la gráfica de los datos y la solución, lo he extraído en una función"""
def printPointsAndSolution(x, y, w, title):
	x1 = []
	x2 = []
	colores = []

	for i in range(len(x)):
		x1.append(x[i][1])
		x2.append(x[i][2])
		if y[i] == 1:
			colores.append('blue')
		else:
			colores.append('orange')


	fig, ax = plt.subplots()
	ax.set(xlabel='Nivel de gris', ylabel='Simetria',
		   title='Dataset y solución ' + title)
	ax.scatter(x1, x2, c=colores)

	valoresDistribuidosX1 = np.linspace(0, 0.6, 1000)
	valoresDistribuidosX2 = []
	for i in range(1000):
		valoresDistribuidosX2.append((-valoresDistribuidosX1[i]*w[1] - w[0])/w[2])

	ax.plot(valoresDistribuidosX1, valoresDistribuidosX2)
	plt.show()
	input("\n--- Pulsar tecla para continuar ---\n")


"""
 (2.5 puntos) Estimar un modelo de regresión lineal a partir de los datos proporcionados de
dichos números (Intensidad promedio, Simetria) usando tanto el algoritmo de la pseudo-
inversa como Gradiente descendente estocástico (SGD). Las etiquetas serán {−1, 1}, una
para cada vector de cada uno de los números. Pintar las soluciones obtenidas junto con los
datos usados en el ajuste. Valorar la bondad del resultado usando Ein y Eout (para Eout cal-
cular las predicciones usando los datos del fichero de test). ( usar Regress_Lin(datos, label)
como llamada para la función (opcional)).

"""

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


eta = 0.01
w = gd(x, y, eta, 500)
printPointsAndSolution(x,y,w, "gradiente decendente")

print ('Bondad del resultado para grad. descendente :\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")

eta = 0.01
n_minibatches = 20
w = sgd(x, y, eta, n_minibatches, 500)

printPointsAndSolution(x,y,w, "gradiente decendente estocástico")

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")


w = pseudoinverse(x, y)

printPointsAndSolution(x,y,w, "Pseudoinversa")

print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")



"""
 En este apartado exploramos como se transforman los errores Ein y Eout cuando au-
mentamos la complejidad del modelo lineal usado. Ahora hacemos uso de la función
simula_unif (N, 2, size) que nos devuelve N coordenadas 2D de puntos uniformemente
muestreados dentro del cuadrado definido por [−size, size] × [−size, size]

"""

print('Ejercicio 2 de regresión lineal\n')

import matplotlib.pyplot as plt
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, size):
	return np.random.uniform(-size, size, N), np.random.uniform(-size, size,N)

"""
Simula la función con ruído, devolviendo la etiqueta y su color en la gráfica
"""

def fRuido(x,y):
	f = (x - 0.2) ** 2 + (y ** 2 - 0.6)
	ruido = np.random.uniform(0, 1)
	if f > 0:
		if ruido > 0.1:
			return "blue", 1
		else:
			return "orange", -1
	else:
		if ruido > 0.1:
			return "orange", -1
		else:
			return "blue", 1

"""Añade un uno y devuelve el array con los dos puntos"""

def xEmpezandoConUno(x,y):
	toReturn = np.empty([len(x), 3])

	for i in range(len(x)):
		punto = toReturn[i]
		punto[0] = 1
		punto[1] = x[i]
		punto[2] = y[i]

	return toReturn

"""Pinta los puntos y devuelve los datasets"""
def get_datasets_experimento(print):
	xpuntos, ypuntos = simula_unif(1000, 1)
	clases = np.empty(1000)
	colores = []

	for i in range(len(xpuntos)):
		color, clase = fRuido(xpuntos[i], ypuntos[i])
		clases[i]= clase
		colores.append(color)

	if print:
		fig, ax = plt.subplots()
		ax.set(xlabel='x', ylabel='y',
			   title='1000 puntos generados aleatoriamente')
		ax.scatter(xpuntos, ypuntos, c=colores)
		plt.show()
		input("\n--- Pulsar tecla para continuar ---\n")

	x = xEmpezandoConUno(xpuntos, ypuntos)

	return x, clases


""" Para imprimir el ajuste a los puntos obtenidos """
def printPointsAndSolutionFor2DPoints(x, y, w):
	x1 = []
	x2 = []
	colores = []

	for i in range(len(x)):
		x1.append(x[i][1])
		x2.append(x[i][2])
		if y[i] == 1:
			colores.append('blue')
		else:
			colores.append('orange')


	fig, ax = plt.subplots()
	ax.set(xlabel='X', ylabel='Y',
		   title='Ajuste al espacio de puntos 2D que hemos generado')
	ax.scatter(x1, x2, c=colores)

	valoresDistribuidosX1 = np.linspace(-0.25, 0.25, 1000)
	valoresDistribuidosX2 = []
	for i in range(1000):
		valoresDistribuidosX2.append((-valoresDistribuidosX1[i]*w[1] - w[0])/w[2])

	ax.plot(valoresDistribuidosX1, valoresDistribuidosX2)
	ax.set_ylim(-1,1)
	plt.show()
	input("\n--- Pulsar tecla para continuar ---\n")

x, y = get_datasets_experimento(True)

eta = 0.01
num_minibatches = 20
maxiter = 500
w = sgd(x, y, eta, num_minibatches, maxiter)

printPointsAndSolutionFor2DPoints(x,y,w)
""" Copiamos x e y para el ajuste de despues"""
xOriginal = x.copy()
yOriginal = y.copy()

print ('Bondad del resultado para gradiente estocástico:\n')
print ("Ein: ", Err(x, y, w))
input("\n--- Pulsar tecla para continuar ---\n")
respuesta = input("¿Quiere ejecutar el experimento de las 1000 iteraciones? (Tarda bastante) [y/n]")

""" Experimento de las mil iteraciones para un modelo lineal"""
if respuesta == "y":
	veces_a_repetir_experimento = 1000
	error_acum_in = 0
	error_acum_out = 0

	for i in range(veces_a_repetir_experimento):
		x, y = get_datasets_experimento(False)
		w = sgd(x, y, eta, num_minibatches, maxiter)
		error_acum_in += Err(x, y, w)
		x_test, y_test = get_datasets_experimento(False)
		error_acum_out += Err(x_test, y_test, w)

	print("El error medio Ein es ", error_acum_in/veces_a_repetir_experimento)
	print("El error medio Eout es ", error_acum_out/veces_a_repetir_experimento)




""" Intentamos aumentar la complejidad del modelo para ajustar mejor a los datos"""

""" Empezamos transformando x """


""" El nuevo X es de 1000 puntos por 6 caracteristicas (las 3 del primer
modelo y 3 nuevas) """

def transformarAlNuevoModelo(xOriginal):
	nuevoX = np.empty([1000, 6], dtype=float)
	for i in range(len(xOriginal)):
		xi = xOriginal[i][1]
		yi = xOriginal[i][2]
		nuevoX[i][0] = 1
		nuevoX[i][1] = xi
		nuevoX[i][2] = yi
		nuevoX[i][3] = xi * yi
		nuevoX[i][4] = xi ** 2
		nuevoX[i][5] = yi ** 2

	return nuevoX


""" Sin optimizar 

def nuevoSgd(x, y, eta, number_of_minibatches, maxiter):
	mini_batches_x = np.array_split(x, number_of_minibatches)
	mini_batches_y = np.array_split(y, number_of_minibatches)
	w = np.array([0, 0, 0,0,0,0], dtype=float)

	index_array = np.arange(0, number_of_minibatches)

	for i in range(maxiter):
		np.random.shuffle(index_array)
		for random_index in index_array:
			xs = mini_batches_x[random_index]
			ys = mini_batches_y[random_index]
			for j in range(len(w)):
				sum = 0
				for n in range(len(xs)):
					guess = w.dot(xs[n])
					sum += (xs[n][j] * (guess - ys[n]))
				w[j] -= (eta * (2*sum)/len(xs))

	return w
	"""

def nuevoSgd(x, y, eta, number_of_minibatches, maxiter):
	mini_batches_x = np.array_split(x, number_of_minibatches)
	mini_batches_y = np.array_split(y, number_of_minibatches)
	w = np.array([0, 0, 0, 0, 0, 0], dtype=float)

	index_array = np.arange(0, number_of_minibatches)

	for i in range(maxiter):
		np.random.shuffle(index_array)
		for random_index in index_array:
			xs = mini_batches_x[random_index]
			ys = mini_batches_y[random_index]
			error_actual = xs.dot(w) - ys
			"""(xs[n][j] * (guess - ys[n])) pero matricial"""
			error_por_x = xs * error_actual.reshape(-1, 1)
			""" Hacemos la media de las columnas nos queda vector 1xd"""
			correcion = np.mean(error_por_x, axis=0)
			""" Corregimos w"""
			w -= (2 * eta * correcion)

	return w

nuevoX = transformarAlNuevoModelo(xOriginal)

wNuevo = nuevoSgd(nuevoX, yOriginal, 0.01, 20, 500)



def nuevoPrintPointsAndSolutionFor2DPoints(x, y, w):
	x1 = []
	x2 = []
	colores = []

	for i in range(len(x)):
		x1.append(x[i][1])
		x2.append(x[i][2])
		if y[i] == 1:
			colores.append('blue')
		else:
			colores.append('orange')


	fig, ax = plt.subplots()
	ax.set(xlabel='X', ylabel='Y',
		   title='Ajuste al espacio de puntos 2D que hemos generado')
	ax.scatter(x1, x2, c=colores)

	valoresDistribuidosX1 = np.linspace(-0.66,1, 1000)
	valoresDistribuidosX2 = []
	valoresDistribuidosX22 = []
	for i in range(1000):
		x1 = valoresDistribuidosX1[i]
		if w[5] == 0:
			x2 = (-x1*w[1] - w[0] - w[4]*x1*x1)/(w[2] + w[3]*x1)
			x22 = x2
		else:
			interior_raiz = (w[2]+w[3]*x1)**2 - (4*w[5]*(x1*w[1]+ w[0]+w[4]*x1*x1))
			x2 = (-np.sqrt(interior_raiz) -w[2] -w[3]*x1) / (2*w[5])
			x22 = (np.sqrt(interior_raiz) -w[2] -w[3]*x1) / (2*w[5])

		valoresDistribuidosX2.append(x2)
		valoresDistribuidosX22.append(x22)

	ax.plot(valoresDistribuidosX1, valoresDistribuidosX2)
	ax.set_ylim(-1,1)
	ax.plot(valoresDistribuidosX1, valoresDistribuidosX22)
	ax.set_ylim(-1,1)
	plt.show()
	input("\n--- Pulsar tecla para continuar ---\n")


nuevoPrintPointsAndSolutionFor2DPoints(nuevoX, yOriginal, wNuevo)
Ein = Err(nuevoX, yOriginal, wNuevo)
print("Ein para el nuevo modelo: ", Ein)
input("\n--- Pulsar tecla para continuar ---\n")


respuesta = input("¿Quiere ejecutar el experimento de las 1000 iteraciones con el nuevo modelo? (Tarda bastante) [y/n]")

""" Experimento de las mil iteraciones para un modelo más complejo"""
if respuesta == "y":
	veces_a_repetir_experimento = 1000
	error_acum_in = 0
	error_acum_out = 0

	for i in range(veces_a_repetir_experimento):
		x, y = get_datasets_experimento(False)
		nuevoX = transformarAlNuevoModelo(x)
		w = nuevoSgd(nuevoX, y, eta, num_minibatches, maxiter)
		error_acum_in += Err(nuevoX, y, w)
		x_test, y_test = get_datasets_experimento(False)
		nuevoXTest = transformarAlNuevoModelo(x_test)
		error_acum_out += Err(nuevoXTest, y_test, w)

	print("El error medio Ein es ", error_acum_in/veces_a_repetir_experimento)
	print("El error medio Eout es ", error_acum_out/veces_a_repetir_experimento)


