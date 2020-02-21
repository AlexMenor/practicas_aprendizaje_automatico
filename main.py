from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()

caracteristicas_en_bruto = data['data']

clases = data['target']

nombre_clases = data['target_names']

clases_y_sus_datos = [{'x': [], 'y': []}, {'x': [], 'y': []}, {'x': [], 'y': []}]

for i in range(0, len(clases)):
    dict = clases_y_sus_datos[clases[i]]
    dict['x'].append(caracteristicas_en_bruto[i][2])
    dict['y'].append(caracteristicas_en_bruto[i][3])

for i in range(0, len(nombre_clases)):
    clases_y_sus_datos[i]['label'] = nombre_clases[i]
    clases_y_sus_datos[i]['color'] = (np.random.rand(), np.random.rand(), np.random.rand())

fig, ax = plt.subplots()

for clase in clases_y_sus_datos:
    ax.scatter(clase['x'], clase['y'], label=clase['label'], color=clase['color'])

ax.legend()
ax.grid(True)
plt.show()






