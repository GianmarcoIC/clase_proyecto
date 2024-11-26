import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from graphviz import Digraph  # Importar Graphviz para graficar la red neuronal

# Configuración Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo de predictivo (Streamlit, Supabase, GitHub y Python)")

# Crear cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")

# Función para obtener datos de la tabla
def get_table_data(table_name):
    """Obtiene todos los datos de una tabla desde Supabase."""
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

# Obtener datos de las tablas
articulos = get_table_data("articulo")
estudiantes = get_table_data("estudiante")
instituciones = get_table_data("institucion")
indizaciones = get_table_data("indizacion")

# Validar datos antes de procesar
if articulos.empty:
    st.error("No hay datos en la tabla 'articulo'. Verifica tu base de datos.")
else:
    # Procesar relaciones entre tablas
    try:
        articulos = articulos.merge(estudiantes, left_on="estudiante_id", right_on="id", suffixes=("", "_estudiante"))
        articulos = articulos.merge(instituciones, left_on="institucion_id", right_on="id", suffixes=("", "_institucion"))
        articulos = articulos.merge(indizaciones, left_on="indizacion_id", right_on="id", suffixes=("", "_indizacion"))
    except KeyError as e:
        st.error(f"Error al unir tablas: {e}")
        st.stop()

    # Procesar los datos
    try:
        articulos['anio_publicacion'] = pd.to_numeric(articulos['anio_publicacion'], errors="coerce")
        datos_modelo = articulos.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')
        
        # Verificar datos procesados
        if datos_modelo.empty:
            st.error("No hay datos suficientes para generar el gráfico.")
            st.stop()
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        st.stop()

    # Modelo predictivo
    try:
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        proximo_anio = articulos['anio_publicacion'].max() + 1
        prediccion = modelo.predict([[proximo_anio]])

        st.write(f"Error cuadrático medio del modelo: {mse:.2f}")
        st.write(f"Predicción para el año {proximo_anio}: {int(prediccion[0])}")
    except Exception as e:
        st.error(f"Error en el modelo predictivo: {e}")
        st.stop()

    # Graficar
    try:
        st.write("Datos procesados para el gráfico:", datos_modelo)  # Mostrar datos procesados
        fig = px.bar(
            datos_modelo,
            x="anio_publicacion",
            y="cantidad_articulos",
            title="Artículos Publicados por Año",
            labels={"anio_publicacion": "Año de Publicación", "cantidad_articulos": "Cantidad de Artículos"}
        )
        st.plotly_chart(fig)
    except ValueError as e:
        st.error(f"Error al generar el gráfico: {e}")

    # Gráfico de la red neuronal
    try:
        st.subheader("Visualización de Red Neuronal")

        nn_graph = Digraph(format="png")
        nn_graph.attr(rankdir="LR")

        # Capas de entrada
        for i in range(1, X.shape[1] + 1):
            nn_graph.node(f"Input_{i}", f"Entrada {i}", shape="circle", style="filled", color="lightblue")

        # Capas ocultas (ejemplo con 3 neuronas)
        for i in range(1, 4):
            nn_graph.node(f"Hidden_{i}", f"Oculta {i}", shape="circle", style="filled", color="lightgreen")

        # Capa de salida
        nn_graph.node("Output", "Salida", shape="circle", style="filled", color="orange")

        # Conexiones
        for i in range(1, X.shape[1] + 1):
            for j in range(1, 4):
                nn_graph.edge(f"Input_{i}", f"Hidden_{j}")

        for i in range(1, 4):
            nn_graph.edge(f"Hidden_{i}", "Output")

        st.graphviz_chart(nn_graph)
    except Exception as e:
        st.error(f"Error al generar el gráfico de la red neuronal: {e}")
