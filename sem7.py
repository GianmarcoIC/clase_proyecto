import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from graphviz import Digraph  # Importar Graphviz para graficar la red neuronal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuración Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

st.image("log_ic-removebg-preview.png", width=200)
st.title("Modelo de predicción con Red Neuronal")

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

    # Modelo de red neuronal
    try:
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        # Normalizar datos para la red neuronal
        X_normalized = (X - X.min()) / (X.max() - X.min())
        y_normalized = (y - y.min()) / (y.max() - y.min())

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

        # Crear modelo de red neuronal
        modelo_nn = Sequential([
            Dense(10, activation='relu', input_dim=1),  # Capa de entrada y oculta
            Dense(10, activation='relu'),  # Segunda capa oculta
            Dense(1, activation='linear')  # Capa de salida
        ])

        modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
        modelo_nn.fit(X_train, y_train, epochs=100, verbose=0)  # Entrenar modelo

        # Predicción
        y_pred_train = modelo_nn.predict(X_train)
        y_pred_test = modelo_nn.predict(X_test)

        # Error del modelo
        mse_nn = mean_squared_error(y_test, y_pred_test)
        st.write(f"Error cuadrático medio del modelo (Red Neuronal): {mse_nn:.4f}")

        # Predicción para el próximo año
        proximo_anio = (articulos['anio_publicacion'].max() + 1 - X.min()) / (X.max() - X.min())  # Normalizar el próximo año
        prediccion_nn = modelo_nn.predict([[proximo_anio]]) * (y.max() - y.min()) + y.min()  # Desnormalizar
        st.write(f"Predicción para el año {articulos['anio_publicacion'].max() + 1}: {int(prediccion_nn[0][0])}")
    except Exception as e:
        st.error(f"Error en el modelo predictivo de red neuronal: {e}")
        st.stop()

    # Mostrar datos y predicciones
    datos_modelo['prediccion'] = modelo_nn.predict(X_normalized) * (y.max() - y.min()) + y.min()
    st.write("Tabla de valores originales y predicciones:")
    st.dataframe(datos_modelo)

    # Graficar barras
    try:
        st.write("Gráfico de barras:")
        fig = px.bar(
            datos_modelo,
            x="anio_publicacion",
            y=["cantidad_articulos", "prediccion"],
            title="Artículos Publicados y Predicción por Año",
            labels={"value": "Cantidad de Artículos", "variable": "Tipo"},
            barmode="group"
        )
        st.plotly_chart(fig)
    except ValueError as e:
        st.error(f"Error al generar el gráfico: {e}")

    # Graficar la red neuronal
    try:
        st.subheader("Visualización de Red Neuronal")

        nn_graph = Digraph(format="png")
        nn_graph.attr(rankdir="LR")

        # Capas de entrada
        nn_graph.node("Input", "Año Normalizado", shape="circle", style="filled", color="lightblue")

        # Capas ocultas (ejemplo con 2 capas y 10 neuronas cada una)
        for i in range(1, 11):
            nn_graph.node(f"Hidden1_{i}", f"Oculta 1-{i}", shape="circle", style="filled", color="lightgreen")
        for i in range(1, 11):
            nn_graph.node(f"Hidden2_{i}", f"Oculta 2-{i}", shape="circle", style="filled", color="lightgreen")

        # Capa de salida
        nn_graph.node("Output", "Predicción", shape="circle", style="filled", color="orange")

        # Conexiones
        for i in range(1, 11):
            nn_graph.edge("Input", f"Hidden1_{i}")
        for i in range(1, 11):
            for j in range(1, 11):
                nn_graph.edge(f"Hidden1_{i}", f"Hidden2_{j}")
        for i in range(1, 11):
            nn_graph.edge(f"Hidden2_{i}", "Output")

        st.graphviz_chart(nn_graph)
    except Exception as e:
        st.error(f"Error al generar el gráfico de la red neuronal: {e}")
