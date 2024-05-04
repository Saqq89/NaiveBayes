# Usamos pandas para cargar los datos desde un archivo de texto, limpiarlos y convertir las columnas categoricas en valores numericos
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# C
# Programacion estructurada y la estructura del programa es secuencial
# Con la biblioteca pandas podemos realizar la manipulación y el análisis de datos.
# Con la biblioteca de sklearn nos proporciona herramientas para el analisis y modelado de datos.
# Cargamos y procesamos los datos del archivo
ruta_archivo = 'C:/Users/Cristian Lopez/Desktop/naives/pythonProject2/tacos.txt'
datos = pd.read_csv(ruta_archivo, delimiter='|', skipinitialspace=True).dropna(axis=1, how='all').dropna(axis=0, how='all')
datos.columns = datos.columns.str.strip()
# D
# Convertimos las columnas categoricas en valores numericos para su procesamiento
for col in datos.columns:
    datos[col] = datos[col].astype('category').cat.codes
# Separamos las caracteristicas (X) y las etiquetas (y)
nombre_clase = 'Clase'
X = datos.drop(nombre_clase, axis=1)
y = datos[nombre_clase]

# Solicitamos al usuario el numero de iteraciones para evaluar el modelo
iteraciones = int(input("Introduce el numero de iteraciones para la evaluacion: "))
precisiones = []


# S
# En cada iteracion, dividimos los datos, entrenamos el modelo Naive Bayes y evaluamos la precision
for i in range(iteraciones):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True)

    # Mostramos datos solo en la primera iteracion para verificar su estructura
    if i == 0:
        print("\nDatos de entrenamiento (primera iteracion):")
        print(X_entrenamiento)
        print("\nDatos de prueba (primera iteracion):")
        print(X_prueba)

    # Entrenamos el modelo Naive Bayes
    modelo = GaussianNB()
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Evaluamos la precision de las predicciones del modelo
    y_predicho = modelo.predict(X_prueba)
    precision = accuracy_score(y_prueba, y_predicho)
    precisiones.append(precision)
    print(f'\nIteracion {i + 1}:')
    print(f'Precision: {precision:.4f}')
# Calculamos y mostramos la precision promedio despues de todas las iteraciones
#fin del codigo 
precision_promedio = sum(precisiones) / iteraciones
print(f'\nPrecision promedio despues de {iteraciones} iteraciones: {precision_promedio:.4f}')
