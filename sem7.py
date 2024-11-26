import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Simular la conexión a la base de datos
def load_data():
    estudiantes = pd.DataFrame([
        [1, 'Juan', 'Pérez', 21, 'VII', 'Ingeniería Informática', 'juan.perez@example.com', '123456789'],
        [2, 'María', 'López', 22, 'VIII', 'Biología', 'maria.lopez@example.com', '987654321'],
        [3, 'Carlos', 'García', 23, 'VI', 'Matemáticas', 'carlos.garcia@example.com', '564738291'],
        [4, 'Ana', 'Torres', 20, 'V', 'Física', 'ana.torres@example.com', '567123456'],
        [5, 'Luis', 'Martínez', 24, 'IX', 'Química', 'luis.martinez@example.com', '789012345']
    ], columns=['id', 'Nombres', 'Apellidos', 'edad', 'ciclo', 'carrera', 'correo', 'telefono'])

    articulos = pd.DataFrame([
        [1, 'Estudio de la inteligencia artificial', '2024-03-15', 2024, '10.1000/xyz123', 1, 1, 1],
        [2, 'Impacto ambiental en zonas urbanas', '2023-11-10', 2023, '10.2000/abc456', 2, 2, 2],
        [3, 'Modelado matemático en biología', '2023-07-08', 2023, '10.3000/def789', 3, 3, 3],
        [4, 'Energía renovable y su crecimiento', '2024-01-05', 2024, '10.4000/ghi101', 4, 4, 4],
        [5, 'Avances en nanotecnología', '2023-05-20', 2023, '10.5000/jkl112', 5, 5, 5]
    ], columns=['id', 'titulo_articulo', 'fecha_publicacion', 'anio_publicacion', 'doi', 'estudiante_id', 'institucion_id', 'indizacion_id'])

    return estudiantes, articulos

# Cargar los datos
estudiantes, articulos = load_data()

# Preprocesamiento
articulos['fecha_publicacion'] = pd.to_datetime(articulos['fecha_publicacion'])
articulos_por_estudiante = articulos.groupby(['estudiante_id', 'anio_publicacion']).size().reset_index(name='articulos_publicados')

# Crear dataset para el modelo
articulos_por_estudiante['articulos_previos'] = articulos_por_estudiante.groupby('estudiante_id')['articulos_publicados'].shift(1).fillna(0)
X = articulos_por_estudiante[['articulos_previos']]
y = articulos_por_estudiante['articulos_publicados']

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Predicción y error
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Predicción por carrera
articulos_por_carrera = articulos.merge(estudiantes[['id', 'carrera']], left_on='estudiante_id', right_on='id')
articulos_por_carrera = articulos_por_carrera.groupby('carrera').size().reset_index(name='articulos_total')
articulos_por_carrera['articulos_previos'] = articulos_por_carrera['articulos_total']
articulos_por_carrera['prediccion'] = model.predict(articulos_por_carrera[['articulos_previos']])

# Visualización en Streamlit
st.title("Modelo Predictivo de Artículos Publicados")
st.write(f"Error cuadrático medio del modelo: {mse:.2f}")

st.subheader("Predicción de Artículos por Carrera")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(articulos_por_carrera['carrera'], articulos_por_carrera['prediccion'], color='skyblue')
ax.set_xlabel("Carrera Profesional")
ax.set_ylabel("Artículos Predichos")
ax.set_title("Predicción de Publicación de Artículos por Carrera")
plt.xticks(rotation=45)
st.pyplot(fig)
