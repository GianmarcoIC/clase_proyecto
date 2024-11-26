import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Configurar Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

# Crear cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")

# Funciones para interactuar con la base de datos
def get_table_data(table_name):
    """Obtiene todos los datos de una tabla."""
    try:
        response = supabase.table(table_name).select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            st.error(f"No se encontraron datos en la tabla {table_name}.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al consultar la tabla {table_name}: {e}")
        return pd.DataFrame()

# Cargar los datos de las tablas
articulos = get_table_data("Articulo")
estudiantes = get_table_data("Estudiante")
instituciones = get_table_data("Institucion")
indizaciones = get_table_data("Indizacion")

# Validar si las tablas tienen datos
if not articulos.empty and not estudiantes.empty:
    # Agregar relaciones entre tablas
    articulos = articulos.merge(estudiantes, left_on="estudiante_id", right_on="id", suffixes=("", "_estudiante"))
    articulos = articulos.merge(instituciones, left_on="institucion_id", right_on="id", suffixes=("", "_institucion"))
    articulos = articulos.merge(indizaciones, left_on="indizacion_id", right_on="id", suffixes=("", "_indizacion"))

    # Procesar y mostrar datos
    articulos['anio_publicacion'] = pd.to_numeric(articulos['anio_publicacion'], errors="coerce")

    # Modelo predictivo
    def modelo_predictivo(articulos):
        datos_modelo = (
            articulos.groupby(['anio_publicacion'])
            .size()
            .reset_index(name='cantidad_articulos')
        )

        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        return modelo, mse

    modelo, mse = modelo_predictivo(articulos)
    proximo_anio = articulos['anio_publicacion'].max() + 1
    prediccion = modelo.predict([[proximo_anio]])

    # Interfaz Streamlit
    st.title("Gestión de Artículos y Predicción")
    st.write(f"Error cuadrático medio del modelo: {mse:.2f}")
    st.write(f"Predicción de artículos para {proximo_anio}: {int(prediccion[0])}")

    # Gráficos
    fig = px.bar(
        articulos,
        x="anio_publicacion",
        y=articulos.groupby('anio_publicacion').size(),
        labels={"y": "Cantidad de Artículos", "anio_publicacion": "Año"},
        title="Artículos Publicados por Año"
    )
    st.plotly_chart(fig)

else:
    st.error("Los datos de artículos o estudiantes no están disponibles. Revisa la base de datos.")
