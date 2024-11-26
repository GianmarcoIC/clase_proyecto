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

# Función para insertar un registro
def insert_record(table_name, data):
    try:
        response = supabase.table(table_name).insert(data).execute()
        st.success(f"Registro insertado en {table_name}.")
        return response.data
    except Exception as e:
        st.error(f"Error al insertar en {table_name}: {e}")

# Función para actualizar un registro
def update_record(table_name, record_id, data):
    try:
        response = supabase.table(table_name).update(data).eq("id", record_id).execute()
        st.success(f"Registro actualizado en {table_name}.")
        return response.data
    except Exception as e:
        st.error(f"Error al actualizar en {table_name}: {e}")

# Función para eliminar un registro
def delete_record(table_name, record_id):
    try:
        response = supabase.table(table_name).delete().eq("id", record_id).execute()
        st.success(f"Registro eliminado de {table_name}.")
        return response.data
    except Exception as e:
        st.error(f"Error al eliminar en {table_name}: {e}")

# Mostrar CRUD en Streamlit
def crud_operations(table_name, data):
    st.subheader(f"Gestión de {table_name.capitalize()}")
    if st.checkbox(f"Mostrar datos de {table_name.capitalize()}"):
        st.dataframe(data)

    # Insertar
    st.subheader(f"Insertar en {table_name.capitalize()}")
    with st.form(f"Insertar_{table_name}"):
        inputs = {col: st.text_input(col) for col in data.columns if col != "id"}
        submitted = st.form_submit_button("Insertar")
        if submitted:
            insert_record(table_name, inputs)

    # Actualizar
    st.subheader(f"Actualizar en {table_name.capitalize()}")
    with st.form(f"Actualizar_{table_name}"):
        record_id = st.text_input("ID del registro a actualizar")
        updates = {col: st.text_input(f"Actualizar {col}") for col in data.columns if col != "id"}
        submitted = st.form_submit_button("Actualizar")
        if submitted:
            update_record(table_name, record_id, updates)

    # Eliminar
    st.subheader(f"Eliminar en {table_name.capitalize()}")
    with st.form(f"Eliminar_{table_name}"):
        record_id = st.text_input("ID del registro a eliminar")
        submitted = st.form_submit_button("Eliminar")
        if submitted:
            delete_record(table_name, record_id)

# Obtener datos de las tablas
articulos = get_table_data("articulo")
estudiantes = get_table_data("estudiante")
instituciones = get_table_data("institucion")
indizaciones = get_table_data("indizacion")

# Mostrar CRUD para cada tabla
crud_operations("articulo", articulos)
crud_operations("estudiante", estudiantes)
crud_operations("institucion", instituciones)
crud_operations("indizacion", indizaciones)

if not articulos.empty:
    # Predicción de publicaciones
    articulos['anio_publicacion'] = pd.to_numeric(articulos['anio_publicacion'], errors="coerce")
    publicaciones_por_año = articulos.groupby('anio_publicacion').size().reset_index(name='publicaciones_totales')

    X = publicaciones_por_año[['anio_publicacion']]
    y = publicaciones_por_año['publicaciones_totales']

    modelo = LinearRegression()
    modelo.fit(X, y)

    años_prediccion = list(range(articulos['anio_publicacion'].max() + 1, articulos['anio_publicacion'].max() + 1 + rango_prediccion))
    predicciones = modelo.predict(pd.DataFrame(años_prediccion, columns=['anio_publicacion']))

    prediccion_df = pd.DataFrame({
        "Año": años_prediccion,
        "Publicaciones Totales (Predicción)": predicciones
    })

    st.write("Predicción de publicaciones totales:")
    st.dataframe(prediccion_df)

    # Gráficos
    fig = px.bar(
        prediccion_df,
        x="Año",
        y="Publicaciones Totales (Predicción)",
        title="Predicción de Publicaciones por Año",
        labels={"Año": "Año", "Publicaciones Totales (Predicción)": "Publicaciones Totales"}
    )
    st.plotly_chart(fig)

    # Red Neuronal
    st.subheader("Red Neuronal")
    nn_graph = Digraph(format="png")
    nn_graph.attr(rankdir="LR")

    # Capas
    for i in range(1, X.shape[1] + 1):
        nn_graph.node(f"Input_{i}", f"Entrada Año {i}", shape="circle", style="filled", color="lightblue")

    for i in range(1, 4):  # Capas ocultas
        nn_graph.node(f"Hidden_{i}", f"Oculta {i}", shape="circle", style="filled", color="lightgreen")

    nn_graph.node("Output", "Salida Predicción", shape="circle", style="filled", color="orange")

    # Conexiones
    for i in range(1, X.shape[1] + 1):
        for j in range(1, 4):
            nn_graph.edge(f"Input_{i}", f"Hidden_{j}")

    for i in range(1, 4):
        nn_graph.edge(f"Hidden_{i}", "Output")

    st.graphviz_chart(nn_graph)
