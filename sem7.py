import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from graphviz import Digraph
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuración Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

st.image("log_ic-removebg-preview.png", width=200)
st.title("CRUD y Modelo Predictivo con Red Neuronal")

# Crear cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Error al conectar con Supabase: {e}")

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

# Función para insertar datos en una tabla
def insert_data(table_name, data):
    try:
        response = supabase.table(table_name).insert(data).execute()
        if response.error:
            st.error(f"Error al insertar datos en {table_name}: {response.error}")
        else:
            st.success(f"Datos insertados correctamente en {table_name}.")
    except Exception as e:
        st.error(f"Error al insertar datos: {e}")

# Función para actualizar datos en una tabla
def update_data(table_name, id_value, data):
    try:
        response = supabase.table(table_name).update(data).eq("id", id_value).execute()
        if response.error:
            st.error(f"Error al actualizar datos en {table_name}: {response.error}")
        else:
            st.success(f"Datos actualizados correctamente en {table_name}.")
    except Exception as e:
        st.error(f"Error al actualizar datos: {e}")

# Función para eliminar datos de una tabla
def delete_data(table_name, id_value):
    try:
        response = supabase.table(table_name).delete().eq("id", id_value).execute()
        if response.error:
            st.error(f"Error al eliminar datos en {table_name}: {response.error}")
        else:
            st.success(f"Datos eliminados correctamente de {table_name}.")
    except Exception as e:
        st.error(f"Error al eliminar datos: {e}")

# CRUD para cada tabla
st.sidebar.title("Operaciones CRUD")
crud_options = st.sidebar.selectbox("Selecciona una tabla para realizar CRUD", ["articulo", "estudiante", "institucion", "indizacion"])

# Leer y mostrar datos
data = get_table_data(crud_options)
st.write(f"Datos actuales en la tabla `{crud_options}`:")
st.dataframe(data)

# Opciones de CRUD
crud_action = st.sidebar.radio("Acción CRUD", ["Crear", "Actualizar", "Eliminar"])

if crud_action == "Crear":
    st.sidebar.write("Inserta nuevos datos:")
    new_data = st.sidebar.text_area("Datos en formato JSON")
    if st.sidebar.button("Insertar"):
        try:
            insert_data(crud_options, eval(new_data))  # Convertir texto en diccionario
        except Exception as e:
            st.error(f"Error en los datos proporcionados: {e}")

elif crud_action == "Actualizar":
    st.sidebar.write("Actualiza datos existentes:")
    record_id = st.sidebar.number_input("ID del registro a actualizar", min_value=0, step=1)
    update_data_json = st.sidebar.text_area("Nuevos datos en formato JSON")
    if st.sidebar.button("Actualizar"):
        try:
            update_data(crud_options, record_id, eval(update_data_json))
        except Exception as e:
            st.error(f"Error en los datos proporcionados: {e}")

elif crud_action == "Eliminar":
    st.sidebar.write("Elimina un registro existente:")
    record_id = st.sidebar.number_input("ID del registro a eliminar", min_value=0, step=1)
    if st.sidebar.button("Eliminar"):
        delete_data(crud_options, record_id)

# Volver a cargar datos después de CRUD
data = get_table_data("articulo")

# Procesar datos para modelo predictivo
if not data.empty:
    try:
        data['anio_publicacion'] = pd.to_numeric(data['anio_publicacion'], errors="coerce")
        datos_modelo = data.groupby(['anio_publicacion']).size().reset_index(name='cantidad_articulos')
        X = datos_modelo[['anio_publicacion']]
        y = datos_modelo['cantidad_articulos']

        # Normalizar datos
        X_normalized = (X - X.min()) / (X.max() - X.min())
        y_normalized = (y - y.min()) / (y.max() - y.min())

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

        # Crear y entrenar red neuronal
        modelo_nn = Sequential([
            Dense(10, activation='relu', input_dim=1),
            Dense(10, activation='relu'),
            Dense(1, activation='linear')
        ])
        modelo_nn.compile(optimizer='adam', loss='mean_squared_error')
        modelo_nn.fit(X_train, y_train, epochs=100, verbose=0)

        # Predicción
        y_pred_test = modelo_nn.predict(X_test)
        mse_nn = mean_squared_error(y_test, y_pred_test)
        st.write(f"Error cuadrático medio del modelo: {mse_nn:.4f}")

        # Gráficos
        datos_modelo['prediccion'] = modelo_nn.predict(X_normalized) * (y.max() - y.min()) + y.min()
        st.write("Tabla con predicciones:")
        st.dataframe(datos_modelo)

        st.write("Gráfico de barras:")
        fig = px.bar(
            datos_modelo,
            x="anio_publicacion",
            y=["cantidad_articulos", "prediccion"],
            title="Artículos Publicados y Predicción",
            labels={"value": "Cantidad de Artículos", "variable": "Tipo"},
            barmode="group"
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error en el modelo predictivo: {e}")

    # Red neuronal con valores
    try:
        st.subheader("Visualización de Red Neuronal con Valores")

        nn_graph = Digraph(format="png")
        nn_graph.attr(rankdir="LR")

        nn_graph.node("Input", "Año de Publicación", shape="circle", style="filled", color="lightblue")
        for i in range(1, 11):
            nn_graph.node(f"Hidden1_{i}", f"Oculta 1-{i}", shape="circle", style="filled", color="lightgreen")
        for i in range(1, 11):
            nn_graph.node(f"Hidden2_{i}", f"Oculta 2-{i}", shape="circle", style="filled", color="lightgreen")
        nn_graph.node("Output", "Cantidad Predicha", shape="circle", style="filled", color="orange")

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
