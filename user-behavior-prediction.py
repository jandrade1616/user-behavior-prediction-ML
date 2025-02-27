#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importación de librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('/datasets/users_behavior.csv')


# In[3]:


# Exploración básica del dataset
print(df.head())
print(df.describe())
print(df.isnull().sum())  # Para comprobar si hay valores faltantes


# In[4]:


# Visualización de la distribución de las variables
df.hist(figsize=(10, 10))
plt.show()

# Visualización de la distribución de la variable objetivo
sns.countplot(x='is_ultra', data=df)
plt.show()

# Matriz de correlación
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# Boxplot para detectar outliers
df.boxplot(figsize=(10, 6))
plt.show()


# In[5]:


features = df.drop(['is_ultra'], axis = 1)
target = df['is_ultra']

features_train, features_temp, target_train, target_temp = train_test_split(
    features, target, test_size=0.4, random_state=12345)  # 60% para entrenamiento

features_valid, features_test, target_valid, target_test = train_test_split(
    features_temp, target_temp, test_size=0.5, random_state=12345)  # 20% validación, 20% prueba


# In[6]:


# Probar diferentes valores para el hiperparámetro max_depth
for depth in range(1, 11):  # max_depth desde 1 hasta 10
    tree_model = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    tree_model.fit(features_train, target_train)  # Entrenar el modelo
    predictions_valid = tree_model.predict(features_valid)  # Predecir en validación
    accuracy = accuracy_score(target_valid, predictions_valid)  # Calcular exactitud
    print(f"Exactitud del Árbol de Decisión con max_depth={depth}: {accuracy}")


# In[7]:


# Probar diferentes valores para n_estimators y max_depth
for est in range(10, 101, 10):  # Cambiar n_estimators entre 10 y 100 en intervalos de 10
    for depth in range(1, 11):  # Cambiar max_depth entre 1 y 10
        forest_model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth)
        forest_model.fit(features_train, target_train)  # Entrenar el modelo
        predictions_valid = forest_model.predict(features_valid)  # Predecir en validación
        accuracy = accuracy_score(target_valid, predictions_valid)  # Calcular exactitud
        print(f"Exactitud con n_estimators={est}, max_depth={depth}: {accuracy}")


# In[8]:


# Probar diferentes valores para el solver y el parámetro de regularización C
for solver in ['liblinear', 'lbfgs']:  # Probar dos solvers populares
    for C_value in [0.1, 1.0, 10.0]:  # Cambiar el parámetro de regularización C
        log_model = LogisticRegression(random_state=12345, solver=solver, C=C_value)
        log_model.fit(features_train, target_train)  # Entrenar el modelo
        predictions_valid = log_model.predict(features_valid)  # Predecir en validación
        accuracy = accuracy_score(target_valid, predictions_valid)  # Calcular exactitud
        print(f"Exactitud con solver={solver}, C={C_value}: {accuracy}")


# In[9]:


# Reentrenar el modelo de Random Forest con los mejores hiperparámetros en el conjunto de entrenamiento completo
best_model = RandomForestClassifier(random_state=12345, n_estimators=40, max_depth=8)

# Entrenar el modelo en todo el conjunto de entrenamiento
best_model.fit(features_train, target_train)

# Predecir en el conjunto de prueba
predictions_test = best_model.predict(features_test)

# Calcular la exactitud en el conjunto de prueba
accuracy_test = accuracy_score(target_test, predictions_test)
print("Exactitud del mejor modelo en el conjunto de prueba:", accuracy_test)


# In[10]:


#Clase mayoritaria en el conjunto de entrenamiento
majority_class = target_train.mode()[0] 

# Predicción trivial 
trivial_predictions = [majority_class] * len(target_test) 

# Exactitud de la predicción trivial
trivial_accuracy = accuracy_score(target_test, trivial_predictions)

print(f"Exactitud de la predicción trivial (prediciendo siempre {majority_class}): {trivial_accuracy}")


#Etiquetas aleatorias del conjunto de entrenamiento
random_target_train = np.random.permutation(target_train)

# Entrenamiento del modelo
random_model = RandomForestClassifier(random_state=12345, n_estimators=50, max_depth=6)
random_model.fit(features_train, random_target_train)

# Predicciones en el conjunto de prueba
random_predictions = random_model.predict(features_test)

# Exactitud del modelo entrenado con etiquetas aleatorias
random_accuracy = accuracy_score(target_test, random_predictions)


print(f"Exactitud del modelo entrenado con etiquetas aleatorias: {random_accuracy}")

# 1. Exactitud del mejor modelo entrenado normalmente
best_model_accuracy = accuracy_score(target_test, predictions_test)

# 2. Exactitud del modelo original para comparar
print(f"Exactitud del mejor modelo (Random Forest) en el conjunto de prueba: {best_model_accuracy}")

# Comparación:
print(f"Predicción trivial: {trivial_accuracy}, Modelo con etiquetas aleatorias: {random_accuracy}, Modelo original: {best_model_accuracy}")



