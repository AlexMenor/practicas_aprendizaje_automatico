# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Alejandro Menor Molinero 13174410X
"""

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
def Err(x,y,w):
	error = 0
	for i in range(len(x)):
		guess = w.dot(x[i])

		error += ((guess - y[i]) ** 2)

	return error / len(x)

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


# Pseudoinversa
def pseudoinverse(x,y):
	pseudoinverse = np.linalg.pinv(x)
	return pseudoinverse.dot(y)



# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

eta = 0.01
w = gd(x, y, eta, 500)

print ('Bondad del resultado para grad. descendente :\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")

eta = 0.01
n_minibatches = 20
w = sgd(x, y, eta, n_minibatches, 500)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")


w = pseudoinverse(x, y)

print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))
input("\n--- Pulsar tecla para continuar ---\n")



print('Ejercicio 2 de regresión lineal\n')

import matplotlib.pyplot as plt
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, size):
	return np.random.uniform(-size, size, N), np.random.uniform(-size, size,N)

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

def xEmpezandoConUno(x,y):
	toReturn = np.empty([len(x), 3])

	for i in range(len(x)):
		punto = toReturn[i]
		punto[0] = 1
		punto[1] = x[i]
		punto[2] = y[i]

	return toReturn

def get_datasets_experimento(print):
	xpuntos, ypuntos = simula_unif(1000, 1)
	clases = []
	colores = []

	for i in range(len(xpuntos)):
		color, clase = fRuido(xpuntos[i], ypuntos[i])
		clases.append(clase)
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



x, y = get_datasets_experimento(True)

eta = 0.01
num_minibatches = 20
maxiter = 500
w = sgd(x, y, eta, num_minibatches, maxiter)

print ('Bondad del resultado para gradiente estocástico:\n')
print ("Ein: ", Err(x, y, w))
input("\n--- Pulsar tecla para continuar ---\n")


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








