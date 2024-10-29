#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Importar bibliotecas necesarias  
import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.preprocessing import LabelEncoder  
from sklearn.feature_selection import SelectKBest, f_regression  

# Paso 1: Cargar el dataset desde la ruta especificada  
ruta_archivo = r'C:\Users\Mariano\Desktop\Analisi de datos\Tarea 3\Guía de actividades y rúbrica de evaluación - Unidad 2 - Tarea 3 - Algoritmos de Aprendizaje Supervi\Anexo 2 - Dataset Vehicle\Car details v3.csv'  
df = pd.read_csv(ruta_archivo)  

# Paso 2: Análisis exploratorio de datos (EDA)  
print("Primeras filas del dataset:")  
print(df.head())  
print("\nResumen estadístico:")  
print(df.describe())  
print("\nInformación general del dataset:")  
print(df.info())  

# Visualización de la distribución de precios  
sns.histplot(df['selling_price'], kde=True)  
plt.title('Distribución de Precios de Venta de Automóviles')  
plt.xlabel('Precio de Venta')  
plt.show()  

# Matriz de correlación para ver relaciones entre variables numéricas únicamente  
numerical_df = df.select_dtypes(include=[np.number])  # Filtra solo las columnas numéricas  
plt.figure(figsize=(10, 8))  
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')  
plt.title('Matriz de Correlación')  
plt.show()  

# Paso 3: Preprocesamiento de datos  
# Convertir variables categóricas en variables numéricas con LabelEncoder  
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']  
label_encoders = {}  
for col in categorical_cols:  
    le = LabelEncoder()  
    df[col] = le.fit_transform(df[col])  
    label_encoders[col] = le  

# Manejo de valores faltantes (si los hay)  
df = df.dropna()  

# Paso 4: Selección de características relevantes para la regresión  
X = df.drop(columns=['selling_price'], errors='ignore')  # Excluir 'selling_price'
y_linear = df['selling_price']  

# Asegurarse de que X solo contiene columnas numéricas
X_numeric = X.select_dtypes(include=[np.number])  

selector = SelectKBest(score_func=f_regression, k='all')  
selector.fit(X_numeric, y_linear)  
X_selected = X_numeric[X_numeric.columns[selector.get_support()]]  

# División de los datos en Train y Test  
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_selected, y_linear, test_size=0.2, random_state=42)  

# Paso 5: Entrenamiento del modelo de Regresión Lineal  
lin_model = LinearRegression()  
lin_model.fit(X_train_linear, y_train_linear)  

# Paso 6: Evaluación del modelo de Regresión Lineal  
y_pred_linear = lin_model.predict(X_test_linear)  
mse = mean_squared_error(y_test_linear, y_pred_linear)  
r2 = r2_score(y_test_linear, y_pred_linear)  

print(f'Mean Squared Error (Regresión Lineal): {mse}')  
print(f'R^2 Score (Regresión Lineal): {r2}')  

# Visualización de los resultados del modelo de Regresión Lineal  
plt.figure(figsize=(10, 6))  
plt.scatter(y_test_linear, y_pred_linear, color='blue', edgecolor='k', alpha=0.6)  
plt.plot([y_test_linear.min(), y_test_linear.max()], [y_test_linear.min(), y_test_linear.max()], color='red', linewidth=2)  
plt.xlabel('Valor Real')  
plt.ylabel('Predicciones')  
plt.title('Predicciones vs Valores Reales (Regresión Lineal)')  
plt.show()  

# Interpretación de resultados  
coef = pd.DataFrame(lin_model.coef_, X_selected.columns, columns=['Coeficiente'])  
print("\nCoeficientes de Regresión Lineal:")  
print(coef)  

# Paso 7: Documentación de resultados  
print("\nInterpretación de Resultados:")  
print("Los coeficientes indican cómo cambia el precio de venta por cada unidad de cambio en las características seleccionadas.")


# In[ ]:




