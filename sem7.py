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

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")

# Funciones para interactuar con la base de datos
def get_articulos():
    """Obtiene todos los artículos desde la base de datos."""
    try:
        response = supabase.table("articulo").select("*").execute()
        if response.status_code == 200 and response.data:
            return pd.DataFrame(response.data)
        else:
            st.error("No se encontraron datos en la tabla articulo o el usuario no tiene permisos adecuados.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al consultar la tabla articulo: {e}")
        return pd.DataFrame()

def get_estudiantes():
    """Obtiene todos los estudiantes desde la base de datos."""
    try:
        response = supabase.table("estudiante").select("*").execute()
        if response.status_code == 200 and response.data:
            return pd.DataFrame(response.data)
        else:
            st.error("No se encontraron datos en la tabla Estudiante o el usuario no tiene permisos adecuados.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al consultar la tabla Estudiante: {e}")
        return pd.DataFrame()

# Comprobación de conexión
if SUPABASE_URL and SUPABASE_KEY:
    try:
        articulos = get_articulos()
        estudiantes = get_estudiantes()
    except Exception as e:
        st.error(f"Error al obtener los datos: {e}")
else:
    st.error("La URL o la clave de Supabase no están configuradas correctamente.")

# Validar datos cargados
if not articulos.empty and not estudiantes.empty:
    # Procesamiento inicial de datos
    articulos['anio_publicacion'] = pd.to_numeric(articulos['anio_publicacion'], errors="coerce")

    # Modelo predictivo
    def modelo_predictivo(articulos):
        carrera_counts = (
            articulos.groupby(['anio_publicacion', 'estudiante_id'])
            .size()
            .reset_index(name='cantidad_articulos')
        )
        carrera_counts = carrera_counts.merge(estudiantes[['id', 'carrera']], left_on='estudiante_id', right_on='id', how='left')

        # Selección de características
        X = carrera_counts[['anio_publicacion']]
        y = carrera_counts['cantidad_articulos']

        # División en datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        return modelo, mse

    modelo, mse = modelo_predictivo(articulos)

    # Predicción para el próximo año
    proximo_anio = articulos['anio_publicacion'].max() + 1
    prediccion = modelo.predict([[proximo_anio]])

    # Interfaz en Streamlit
    st.title("Gestión de Artículos y Predicción")
    st.write(f"Error cuadrático medio del modelo: {mse:.2f}")
    st.write(f"Predicción de artículos para {proximo_anio}: {int(prediccion[0])}")

    # Mostrar los datos en un gráfico
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
