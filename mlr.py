import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Carga datos
dataset = pd.read_csv('cleaned_nutrition_dataset.csv')

# Separa las variables independientes (X) de la variable dependiente (y).
X = dataset.iloc[:,[0,1,2,3,4,6,8,9,10,11]] # Características
y = dataset.iloc[:,7] # Valor Calórico

# Divide dateset en conjuntos: entrenamiento 80% & prueba 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrena el modelo de regresión lineal múltiple
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predice los resultados del conjunto de prueba
y_pred = regressor.predict(X_test)

# Configura la salida de NumPy para mostrar números con dos decimales.
np.set_printoptions(precision=2)

# Concatena las predicciones (y_pred) y los valores reales (y_test) en columnas.
resultados = np.concatenate((
    y_pred.reshape(-1, 1),
    y_test.values.reshape(-1, 1)
), axis=1)

print("Predicciones vs Valores Reales:\n", resultados)

# Visualiza predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores reales')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones')
plt.title('Predicciones vs Valores reales')
plt.xlabel('Índice de muestra')
plt.ylabel('Valor Calórico')
plt.legend()
plt.grid()
plt.show()

# Evaluación
import statsmodels.api as sm
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()
print(model_sm.summary()) # P Valor

from sklearn.metrics import r2_score, root_mean_squared_error
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))