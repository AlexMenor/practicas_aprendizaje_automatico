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
def sign(error):
	if error > 0:
		return 1
	else:
		return-1

def Err(x,y,w):
	error = 0
	for i in range(len(x)):
		guess = 0
		for j in range(len(x[i])):
			guess += w[j] * x[i][j]

		if sign(guess) != y[i]:
			error += 1


	return error / len(x)

def h(x, w):
	guess = 0
	for i in range (len(x)):
		guess += x[i]*w[i]
	return guess

# Gradiente Descendente
def gd(x, y, eta):
	w = np.array([0, 0, 0])
	for i in range(1000):
		for j in range(len(w)):
			sum = 0
			for n in range(len(x)):
				guess = sign(h(x[n], w))
				sum += (x[n][j] * (guess - y[n]))
			w[j] -= eta * sum

	return w
# Gradiente Descendente Estocastico
def sgd(x, y, eta, number_of_minibatches):
	mini_batches_x = np.array_split(x, number_of_minibatches)
	mini_batches_y = np.array_split(y, number_of_minibatches)
	w = np.array([0, 0, 0])

	for xs, ys in zip(mini_batches_x, mini_batches_y):
		for j in range(len(w)):
			sum = 0
			for n in range(len(xs)):
				guess = sign(h(xs[n], w))
				sum += (xs[n][j] * (guess - ys[n]))
			w[j] -= eta * sum

	return w

# Pseudoinversa	
def pseudoinverse():
    pass


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

eta = 0.01
w = gd(x, y, eta)

print ('Bondad del resultado para grad. descendente :\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

eta = 0.01
n_minibatches = 30
w = sgd(x, y, eta, n_minibatches)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

"""

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...
"""


