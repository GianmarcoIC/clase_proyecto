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
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Funciones para interactuar con la base de datos
def get_articulos():
    response = supabase.table('Articulo').select('*').execute()
    return pd.DataFrame(response.data)

def get_estudiantes():
    response = supabase.table('Estudiante').select('*').execute()
    return pd.DataFrame(response.data)

def add_articulo(titulo, fecha, anio, doi, estudiante_id, institucion_id, indizacion_id):
    supabase.table('Articulo').insert({
        "titulo_articulo": titulo,
        "fecha_publicacion": fecha,
        "anio_publicacion": anio,
        "doi": doi,
        "estudiante_id": estudiante_id,
        "institucion_id": institucion_id,
        "indizacion_id": indizacion_id
    }).execute()

def delete_articulo(articulo_id):
    supabase.table('Articulo').delete().eq('id', articulo_id).execute()

# Cargar datos iniciales
articulos = get_articulos()
estudiantes = get_estudiantes()

# Procesamiento inicial de datos
articulos['anio_publicacion'] = pd.to_numeric(articulos['anio_publicacion'])

# Modelo predictivo
def modelo_predictivo(articulos):
    carrera_counts = (
        articulos.groupby(['anio_publicacion', 'estudiante_id'])
        .size()
        .reset_index(name='cantidad_articulos')
    )
    carrera_counts = carrera_counts.merge(estudiantes[['id', 'carrera']], left_on='estudiante_id', right_on='id')
    
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

# CRUD
menu = ["Ver Artículos", "Agregar Artículo", "Eliminar Artículo"]
choice = st.sidebar.selectbox("Menú", menu)

if choice == "Ver Artículos":
    st.subheader("Artículos Publicados")
    st.dataframe(articulos)

elif choice == "Agregar Artículo":
    st.subheader("Agregar un Nuevo Artículo")
    titulo = st.text_input("Título")
    fecha = st.date_input("Fecha de Publicación")
    anio = st.number_input("Año de Publicación", min_value=2000, max_value=proximo_anio)
    doi = st.text_input("DOI")
    estudiante_id = st.selectbox("Estudiante", estudiantes['id'])
    institucion_id = st.number_input("ID Institución")
    indizacion_id = st.number_input("ID Indización")
    if st.button("Agregar"):
        add_articulo(titulo, fecha, anio, doi, estudiante_id, institucion_id, indizacion_id)
        st.success("Artículo agregado exitosamente")

elif choice == "Eliminar Artículo":
    st.subheader("Eliminar Artículo")
    articulo_id = st.number_input("ID del Artículo a eliminar", min_value=1)
    if st.button("Eliminar"):
        delete_articulo(articulo_id)
        st.success("Artículo eliminado exitosamente")
