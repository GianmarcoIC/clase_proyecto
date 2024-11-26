import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from graphviz import Digraph

# Configuración Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo Predictivo Basado en Artículos, Estudiantes, Indización e Instituciones")

# Crear cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")

# Configuración del rango de predicción
st.sidebar.title("Configuración de Predicción")
rango_prediccion = st.sidebar.number_input("Rango de predicción (años)", min_value=1, value=3, step=1)

# Función para obtener datos de una tabla
def get_table_data(table_name):
    try:
        response = supabase.table(table_name).select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            st.warning(f"La tabla {table_name} está vacía.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al consultar la tabla {table_name}: {e}")
        return pd.DataFrame()

# Datos principales
articulos = get_table_data("articulo")
estudiantes = get_table_data("estudiante")
instituciones = get_table_data("institucion")
indizaciones = get_table_data("indizacion")

if not articulos.empty and not estudiantes.empty and not instituciones.empty and not indizaciones.empty:
    # Procesar datos
    articulos['anio_publicacion'] = pd.to_numeric(articulos['anio_publicacion'], errors="coerce")
    publicaciones_por_año = articulos.groupby('anio_publicacion').size().reset_index(name='publicaciones_totales')

    # Predicción de artículos
    try:
        X_articulos = publicaciones_por_año[['anio_publicacion']]
        y_articulos = publicaciones_por_año['publicaciones_totales']

        modelo_articulos = LinearRegression()
        modelo_articulos.fit(X_articulos, y_articulos)

        años_prediccion = list(range(articulos['anio_publicacion'].max() + 1, articulos['anio_publicacion'].max() + 1 + rango_prediccion))
        predicciones_articulos = modelo_articulos.predict(pd.DataFrame(años_prediccion, columns=['anio_publicacion']))

        prediccion_articulos_df = pd.DataFrame({
            "Año": años_prediccion,
            "Publicaciones Totales (Predicción)": predicciones_articulos
        })

        st.write("Predicción de publicaciones totales:")
        st.dataframe(prediccion_articulos_df)

        fig_articulos = px.bar(
            prediccion_articulos_df,
            x="Año",
            y="Publicaciones Totales (Predicción)",
            title="Predicción de Publicaciones por Año",
            labels={"Año": "Año", "Publicaciones Totales (Predicción)": "Publicaciones Totales"}
        )
        st.plotly_chart(fig_articulos)

    except Exception as e:
        st.error(f"Error al predecir artículos: {e}")

    # Predicción de indización
    try:
        indizaciones_por_año = articulos.merge(indizaciones, left_on="indizacion_id", right_on="id")
        niveles = ["Q1", "Q2", "Q3", "Q4"]

        predicciones_indizacion = {}
        for nivel in niveles:
            nivel_data = indizaciones_por_año[indizaciones_por_año['nivel'] == nivel].groupby('anio_publicacion').size()
            nivel_data = nivel_data.reindex(range(articulos['anio_publicacion'].min(), articulos['anio_publicacion'].max() + 1), fill_value=0).reset_index()
            nivel_data.columns = ['anio_publicacion', 'cantidad']
            modelo_nivel = LinearRegression()
            modelo_nivel.fit(nivel_data[['anio_publicacion']], nivel_data['cantidad'])
            predicciones_indizacion[nivel] = modelo_nivel.predict(pd.DataFrame(años_prediccion, columns=['anio_publicacion']))

        predicciones_indizacion_df = pd.DataFrame(predicciones_indizacion, index=años_prediccion)
        predicciones_indizacion_df.index.name = "Año"
        st.write("Predicción de Indización por Nivel:")
        st.dataframe(predicciones_indizacion_df)

        fig_indizaciones = px.line(
            predicciones_indizacion_df,
            title="Predicción de Publicaciones por Nivel de Indización",
            labels={"value": "Cantidad", "variable": "Nivel"}
        )
        st.plotly_chart(fig_indizaciones)

    except Exception as e:
        st.error(f"Error al predecir indizaciones: {e}")

    # Predicción de estudiantes
    try:
        estudiantes_por_año = articulos.groupby('anio_publicacion')['estudiante_id'].nunique().reset_index()
        estudiantes_por_año.columns = ['anio_publicacion', 'estudiantes_unicos']

        modelo_estudiantes = LinearRegression()
        modelo_estudiantes.fit(estudiantes_por_año[['anio_publicacion']], estudiantes_por_año['estudiantes_unicos'])

        predicciones_estudiantes = modelo_estudiantes.predict(pd.DataFrame(años_prediccion, columns=['anio_publicacion']))
        prediccion_estudiantes_df = pd.DataFrame({
            "Año": años_prediccion,
            "Estudiantes Publicadores (Predicción)": predicciones_estudiantes
        })

        st.write("Predicción de estudiantes publicadores:")
        st.dataframe(prediccion_estudiantes_df)

        fig_estudiantes = px.line(
            prediccion_estudiantes_df,
            x="Año",
            y="Estudiantes Publicadores (Predicción)",
            title="Predicción de Estudiantes por Año",
            labels={"Año": "Año", "Estudiantes Publicadores (Predicción)": "Cantidad"}
        )
        st.plotly_chart(fig_estudiantes)

    except Exception as e:
        st.error(f"Error al predecir estudiantes: {e}")

    # Predicción de instituciones
    try:
        publicaciones_instituciones = articulos.merge(instituciones, left_on="institucion_id", right_on="id")
        instituciones_predicciones = publicaciones_instituciones.groupby(['anio_publicacion', 'nombre']).size().reset_index(name='cantidad')

        predicciones_instituciones_df = instituciones_predicciones.groupby('nombre').apply(
            lambda grupo: grupo.set_index('anio_publicacion').reindex(range(articulos['anio_publicacion'].min(), articulos['anio_publicacion'].max() + 1), fill_value=0)
        ).reset_index()

        st.write("Predicción de publicaciones por institución:")
        st.dataframe(predicciones_instituciones_df)

        fig_instituciones = px.bar(
            predicciones_instituciones_df,
            x="anio_publicacion",
            y="cantidad",
            color="nombre",
            title="Predicción por Institución",
            labels={"anio_publicacion": "Año", "cantidad": "Cantidad de Publicaciones", "nombre": "Institución"}
        )
        st.plotly_chart(fig_instituciones)

    except Exception as e:
        st.error(f"Error al predecir instituciones: {e}")
