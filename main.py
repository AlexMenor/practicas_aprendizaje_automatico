# PRÁCTICA 0 -- APRENDIZAJE AUTOMÁTICO
# ALEJANDRO MENOR MOLINERO

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()

caracteristicas_en_bruto = data['data']

clases = data['target']

nombre_clases = data['target_names']

# PARTE 1

clases_y_sus_datos = [{'x': [], 'y':[]}, {'x': [], 'y':[]}, {'x': [], 'y':[]}]

for i in range(0, len(clases)):
    dict = clases_y_sus_datos[clases[i]]
    dict['x'].append(caracteristicas_en_bruto[i][2])
    dict['y'].append(caracteristicas_en_bruto[i][3])

for i in range(0, len(nombre_clases)):
    clases_y_sus_datos[i]['label'] = nombre_clases[i]

clases_y_sus_datos[0]['color'] = 'pink'
clases_y_sus_datos[1]['color'] = 'blue'
clases_y_sus_datos[2]['color'] = 'green'

fig, (ax1, ax2) = plt.subplots(1, 2)

for clase in clases_y_sus_datos:
    ax1.scatter(clase['x'], clase['y'], label=clase['label'], color=clase['color'])

ax1.legend()
ax1.grid(True)

# PARTE 2

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(caracteristicas_en_bruto, clases, test_size=0.2, train_size=0.8, stratify=clases)

## Comprobación


print("Parte 2")

for i in range (0, 3):
    print(f"Proporción de valores de la clase {i}")
    print(np.count_nonzero(y_train == i) / len(y_train))

print(f"Tamaño de muestras de entrenamiento: {len(y_train)} ({len(y_train)/len(clases) * 100} %)")
print(f"Tamaño de muestras de test: {len(y_test)} ({len(y_test)/len(clases) * 100} %)")


# PARTE 3

samples = np.linspace(0, 2 * np.pi, 100)

sin_arr = np.sin(samples)
cos_arr = np.cos(samples)

ax2.plot(samples, sin_arr, color="black", linestyle="dashed", label="sin(x)")
ax2.plot(samples, cos_arr, color="blue", linestyle="dashed", label="cos(x)")
ax2.plot(samples, np.add(sin_arr, cos_arr), color="red", linestyle="dashed", label="sin(x) + cos(x)")
ax2.legend()

plt.show()
