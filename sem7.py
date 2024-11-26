import streamlit as st
import pandas as pd
from supabase import create_client, Client

# Configurar Supabase
SUPABASE_URL = "https://msjtvyvvcsnmoblkpjbz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1zanR2eXZ2Y3NubW9ibGtwamJ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzIwNTk2MDQsImV4cCI6MjA0NzYzNTYwNH0.QY1WtnONQ9mcXELSeG_60Z3HON9DxSZt31_o-JFej2k"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Funciones CRUD
def get_students():
    response = supabase.table('estudiante').select('*').execute()
    return response.data

def count_students():
    response = supabase.table('estudiante').select('*', count='exact').execute()
    return response.count

def add_student(name, age, ciclo, carrera, correo, telefono):
    supabase.table('estudiante').insert({
        "nombres": name,
        "edad": age,
        "ciclo": ciclo,
        "carrera": carrera,
        "correo": correo,
        "telefono": telefono
    }).execute()

def update_student(student_id, name, age, ciclo, carrera, correo, telefono):
    supabase.table('estudiante').update({
        "nombres": name,
        "edad": age,
        "ciclo": ciclo,
        "carrera": carrera,
        "correo": correo,
        "telefono": telefono
    }).eq("id", student_id).execute()

def delete_student(student_id):
    supabase.table('estudiante').delete().eq("id", student_id).execute()

def paginate_dataframe(df, page_size):
    """Divide el DataFrame en páginas de un tamaño dado."""
    num_pages = (len(df) + page_size - 1) // page_size
    for page in range(num_pages):
        start_idx = page * page_size
        end_idx = min((page + 1) * page_size, len(df))
        yield df.iloc[start_idx:end_idx]

# Interfaz Streamlit
st.image("log_ic-removebg-preview.png", width=200)
st.title("CRUD Python - Instituto Continental IDL3")

menu = ["Ver", "Agregar", "Actualizar", "Eliminar"]
choice = st.sidebar.selectbox("Menú", menu)

if choice == "Ver":
    st.subheader("Lista de estudiantes")
    students = get_students()
    student_count = count_students()
    st.write(f"Cantidad total de estudiantes: {student_count}")
    df_students = pd.DataFrame(students)
    if not df_students.empty:
        page_size = 5
        df_students['#'] = df_students.index + 1
        num_pages = (len(df_students) + page_size - 1) // page_size
        page_number = st.number_input("Selecciona la página:", min_value=1, max_value=num_pages, value=1)
        for page_df in paginate_dataframe(df_students, page_size):
            if page_number == (df_students.index[page_df.index[-1]] // page_size + 1):
                st.dataframe(page_df[['#', 'id', 'nombres', 'edad', 'ciclo', 'carrera', 'correo', 'telefono']])
                break
    else:
        st.info("No hay estudiantes registrados")

elif choice == "Agregar":
    st.subheader("Agregar Estudiante")
    name = st.text_input("Nombre")
    age = st.number_input("Edad", min_value=1, max_value=100)
    ciclo = st.text_input("Ciclo")
    carrera = st.text_input("Carrera")
    correo = st.text_input("Correo")
    telefono = st.text_input("Teléfono")
    if st.button("Agregar") and name and correo and telefono:
        add_student(name, age, ciclo, carrera, correo, telefono)
        st.success("Estudiante agregado exitosamente")

elif choice == "Actualizar":
    st.subheader("Actualizar Estudiante")
    student_id = st.number_input("ID del estudiante", min_value=1)
    name = st.text_input("Nuevo Nombre")
    age = st.number_input("Nueva Edad", min_value=1, max_value=100)
    ciclo = st.text_input("Nuevo Ciclo")
    carrera = st.text_input("Nueva Carrera")
    correo = st.text_input("Nuevo Correo")
    telefono = st.text_input("Nuevo Teléfono")
    if st.button("Actualizar") and student_id:
        update_student(student_id, name, age, ciclo, carrera, correo, telefono)
        st.success("Estudiante actualizado exitosamente")

elif choice == "Eliminar":
    st.subheader("Eliminar Estudiante")
    student_id = st.number_input("ID del estudiante", min_value=1)
    if st.button("Eliminar") and student_id:
        delete_student(student_id)
        st.success("Estudiante eliminado exitosamente")
