#!/usr/bin/env python
# coding: utf-8

# # Optical Recognition of Handwritten Digits
# ## El problema
# Tenemos un problema de clasificación con 10 clases. Los datos vienen de una universidad de Turquía en la que procesaron dígitos escritos a mano como mapas de bits.
# ## El data set
# Los mapas de bits tienen 32 x 32 bits. Por suerte, los datos no son 32 x 32 columnas de booleanos. En cambio, estos investigadores, dividieron todo el mapa de bits en 64 cuadros de 4x4 bits e hicieron de cada cuadro una columna. Por tanto, cada uno de esos atributos, contiene la cuenta de bits "true" que hay en ese cuadro, por lo tanto, están entre 0 y 16. Es así como consiguieron reducir las dimensiones del problema.
# ## Elementos
# - X: 64 columnas, cada una tiene un número entre 0 y 16 que representa la cuenta de bits activos en un cuadro del mapa.
# - Y: La etiqueta tiene como rango 0-9 (que son todos los dígitos que contempla el sistema numérico decimal).
# - Función objetivo: Dados los bits que contiene cada uno de los cuadros del mapa, predecir de que dígito se trata.
# ## Archivo de datos
# En este caso no tenemos ni si quiera que hacer nosotros la separación entre datos de entrenamiento / test. Tenemos dos archivos CSV sin cabecera: 
# - optdigits.tra: Datos de training.
# - optdigits.tes: Datos de test.

# ## Funciones a utilizar
# De nuevo, empiezo con combinaciones lineales sin transformación ninguna. Creo que no hace falta (y he comprobado más tarde estar en lo cierto) y añadir complejidad innecesaria a nuestro modelo provocaría aumentar el error fuera de la muestra al sacrificar varianza por sesgo.

# ## Datos perdidos
# Según el fichero "optdigits.name", que da información sobre el data set, no hay datos perdidos. Vamos sin embargo a comprobar rápidamente que no hay datos perdidos y que hemos hecho correctamente la lectura.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Leemos los CSV con header=None para que no cuente la primera línea como cabecera
raw_data_training = pd.read_csv('./datos/optdigits.tra', header=None)
raw_data_test = pd.read_csv('./datos/optdigits.tes', header=None)

# Miramos a ver si hay algún dato perdido
print('Hay datos nulos en training:',raw_data_training.isna().any().any())
print('Hay datos nulos en test: ', raw_data_test.isna().any().any())

input('\nPulse cualquier tecla para continuar\n')

# ## Muestreo estratificado
# Podemos ver que tanto los datos de training como los de entrenamiento están estratificados por dígitos. Es decir, hay un equilibrio de ocurrencias de cada dígito. De esta forma, no hay falta de información sobre ninguno de ellos.

# In[2]:



# Imprimimos una representación gráfica del número de ocurrencias de cada dígito.

pd.value_counts(raw_data_training[64]).plot.bar(title="Conteo de dígitos en los datos de training").plot()
plt.show()
input('\nPulse cualquier tecla para continuar\n')


# In[3]:



# Imprimimos una representación gráfica del número de ocurrencias de cada dígito.

pd.value_counts(raw_data_test[64]).plot.bar(title="Conteo de dígitos en los datos de test").plot()
plt.show()
input('\nPulse cualquier tecla para continuar\n')


# In[4]:


# Partimos los datos en X Y, siendo 64 la columna que contiene la etiqueta o target

X_train = raw_data_training.drop(64, axis=1)
Y_train = raw_data_training[64]

print('X_train shape:', X_train.shape, 'Y_train shape:', Y_train.shape)

X_test = raw_data_test.drop(64, axis=1)
Y_test = raw_data_test[64]

print('X_test shape:', X_test.shape, 'Y_test shape:', Y_test.shape)

input('\nPulse cualquier tecla para continuar\n')


# ## Métrica de ajuste
# Hablaré en detalle de la función de pérdida de cada modelo que utilice. En cuanto a la que he considerado en validación y Etest, considero accuary (el porcentaje de aciertos) pragmática y representativa.

# ## Validación cruzada
# En este caso tenemos una validación cruzada un poco más sofisticada que en el caso de regresión:
# - Utilizamos "StratifiedKFold" para que parta los datos de training en 10 partes estratificadas, con aproximadamente la misma cantidad de ocurrencias de cada dígito.
# - Utilizamos pipeline para encadenar también un StandardScaler que estandariza el rango de las columnas a una distribución Gaussiana con media 0 y varianza 1. Aunque realmente ya estaban todas las columnas regularizadas a un rango 0-16, se considera que normalizarlo a 0-1 o a media 0 varianza 1 es beneficioso para los modelos (incluso requerido por algunos).
# z = (x - u) / s  (Siendo u la media de la columna y s la desviación tipica) Para cada dato (x).
# - Utilizamos como métrica la precisión (porcentaje de aciertos)

# In[5]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Utilizamos cross_val_score de sklearn que hace la validación cruzada por nosotros.
# Devuelve el valor de cada validación (en este caso 10)

def validate_models(model_strings, models, X_train, Y_train):
    validation_results = {'accuracy':[], 'std_dev_error': []}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for model in models:
        pipeline = make_pipeline(StandardScaler(), model)
        results = cross_val_score(pipeline, X_train, Y_train, scoring = "accuracy", cv=cv)
        
        # Hacemos la media y desviación típica de las 10 validaciones
        
        validation_results['accuracy'].append(np.mean(np.abs(results)))
        validation_results['std_dev_error'].append(np.std(np.abs(results)))
        
        # Devolvemos una tabla con los modelos y sus métricas
        
    return pd.DataFrame(data=validation_results, index=model_strings)


# ## Regresión logística
# - A diferencia de la regresión logística que hemos trabajado en otras prácticas, en esta contamos con regularización por normal l2, minimizando la siguiente función:
# $\min_{w, c} \frac{1}{2}w^T w + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1)$
# - Además, controlamos la regularización con el parámetro C (valores más pequeños, mas regularización).
# - Como función de pérdida, como el parámetro multi_class a multinomial, se utiliza cross-entropy.
# 
# Se calcula como $\sum_{c=1}^My_{o,c}\log(p_{o,c})$, donde:
# - $M$ es el núero de clases (en este caso 10)
# - $Y$: 0 o 1 dependiendo si $c$ es la clasificación correcta del dato $o$
# - $p$: Probabilidad predicha de que el dato $o$ pertenezca a la clase $c$
# 
# También se denomina a veces perdida logarítmica por su curva:
# ![log-loss](https://ml-cheatsheet.readthedocs.io/en/latest/_images/cross_entropy.png)
# 

# In[6]:



# Modelo de regresión logística

from sklearn.linear_model import LogisticRegression
models = [
    LogisticRegression(multi_class='multinomial', C=0.5),
    LogisticRegression(multi_class='multinomial', C=1.0),
    LogisticRegression(multi_class='multinomial', C=1.5),
]

model_strings = [
    'Logistic Regression C = 0.5',
    'Logistic Regression C = 1.0',
    'Logistic Regression C = 1.5',
]

print(validate_models(model_strings, models, X_train, Y_train))
input('\nPulse cualquier tecla para continuar\n')


# ## Perceptrón
# Un viejo conocido, en este caso también tiene opción de regularización. He probado, como se puede ver más abajo, con y sin ella (norma l2).
# 
# - Itera por todos los datos.
# - Si está bien situado para el dato dado, no cambia.
# - Si no lo está, se corrige.
# - Si no cambia en una pasada completa o llega a las iteraciones máximas, para.

# In[7]:



# Modelo Perceptrón

from sklearn.linear_model import Perceptron
models = [
    Perceptron(random_state=1),
    Perceptron(alpha=0.0001, penalty='l2',random_state=1),
    Perceptron(alpha=0.00025, penalty='l2',random_state=1),
    Perceptron(alpha=0.0004, penalty='l2',random_state=1),
    
]

model_strings = [
    'Perceptron sin regularización',
    'Perceptron alpha = 0.0001',
    'Perceptron alpha = 0.00025',
    'Perceptron alpha = 0.0004',
  
]

print(validate_models(model_strings, models, X_train, Y_train))
input('\nPulse cualquier tecla para continuar\n')


# ## Ridge Classifier
# - Se optimiza la misma función que hemos visto para el problema de regresión. 
# - Se menciona en la documentación que puede ser más rápido para problemas multiclases (como este) en comparación regresión logística ya que solo se computa la matriz de proyección una vez. $(X^T X)^{-1} X^T$
# - De nuevo, he probado con y sin regularización. 
# - Utilizo como algoritmo el de la descomposición en valores singulares, al igual que OLS.

# In[8]:



# Modelo Ridge Classifier

from sklearn.linear_model import RidgeClassifier
models = [
    RidgeClassifier(alpha=0, solver='svd'),
    RidgeClassifier(alpha=1.0, solver='svd')
    
]

model_strings = [
    'Ridge alpha = 0',
    'Ridge alpha = 1.0',
]

print(validate_models(model_strings, models, X_train, Y_train))
input('\nPulse cualquier tecla para continuar\n')


# ## Elección del mejor modelo
# De nuevo, gracias a la validación podemos escoger (con seguridad de que no ha habido overfitting y de que Eval es un indicador no sesgado del error fuera de la muestra) el modelo que mejor resultados nos ha dado.
# - Ha sido Regresión Logística con C=1.5 para regularizar.
# - A diferencia del anterior problema, ahora si que le aplicamos un StandardScaler (igual que hemos hecho al validar) a los datos antes de entrenar con ellos.

# In[9]:



# Normalizamos los datos de training antes de usarlos

stdScaler = StandardScaler()
stdScaler.fit(X_train)
X_train_scaled = stdScaler.transform(X_train)

# Entrenamos con ellos

final_model = LogisticRegression(multi_class='multinomial', C=1.5)
final_model.fit(X_train_scaled, Y_train)


# ## Usamos Etest para ver cómo de bueno es nuestro modelo final con datos nuevos.
# - De nuevo normalizamos.
# - Predecimos.
# - Medimos el porcentaje de aciertos.

# In[10]:


from sklearn.metrics import accuracy_score

# Normalizamos los datos de test antes de usarlos.

stdScaler = StandardScaler()
stdScaler.fit(X_test)
X_test_scaled = stdScaler.transform(X_test)

# Predecimos 


# Medimos el porcentaje de aciertos

predictions = final_model.predict(X_test_scaled)
accuracy_test = accuracy_score(Y_test, predictions)
print('Accuracy in test set: ', accuracy_test)
input('\nPulse cualquier tecla para continuar\n')


# ## Analizando la confusion-matrix
# - El número más grande de confusiones se ha dado clasificando unos como ochos (ha ocurrido 9 veces)
# - También es destacable la confusión de clasificar nueves como sietes y nueves como ochos (6 y 7 veces respectivamente)

# In[11]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(Y_test, predictions))
input('\nPulse cualquier tecla para continuar\n')


# ## Analizando classifcation report
# - Acorde a lo que hemos visto en la confusion matrix, el nueve ha sido el dígito que peor se ha clasificado (solo se ha hecho bien el 88% de las veces).
# - El cero no se ha clasificado mal ninguna vez.

# In[12]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))
input('\nPulse cualquier tecla para continuar\n')


# ## Estimación de Eout
# Ya he hablado del modelo que he elegido y hemos usado el conjunto test para probarlo con datos "nuevos".
# Vamos ahora a usar la misma cota que en el ejercicio de regresión: $E_{out}(g) <= E_{test}(g) + \sqrt(\frac{1}{2N}ln\frac{2}{\alpha})$
# 
# Para poder asegurar al 95% de confianza que Eout va a ser menor o igual a 

# In[13]:


N = len(X_test)
alpha = 0.05
etest = 1 - accuracy_test

eout_bound = etest + np.sqrt((1 / (2*N)) * np.log(2 / alpha))

print('Eout(g) <= ', eout_bound)

input('\nPulse cualquier tecla para continuar\n')




