import streamlit as st
from supabase import create_client, Client

# Configurar Supabase
SUPABASE_URL = "https://peioqwvlxrgujotcuazt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBlaW9xd3ZseHJndWpvdGN1YXp0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQwMzM0MDUsImV4cCI6MjAzOTYwOTQwNX0.fLmClBVIcVGr_iKYTw79kPJUb12Iem7beooWfesNiXE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_students():
    response = supabase.table('students').select('*').execute()
    return response.data

def count_students():
    response = supabase.table('students').select('*', count='exact').execute()
    return response.count

def add_student(name, age):
    supabase.table('students').insert({"name": name, "age": age}).execute()

def update_student(student_id, name, age):
    supabase.table('students').update({"name": name, "age": age}).eq("id", student_id).execute()

def delete_student(student_id):
    supabase.table('students').delete().eq("id", student_id).execute()
    
st.image("log_ic-removebg-preview.png", width=150)
st.title("Proyecto Prodcutivo IDL3")

menu = ["Ver", "Agregar", "Actualizar", "Eliminar"]
choice = st.sidebar.selectbox("Menú", menu)

if choice == "Ver":
    st.subheader("Lista de estudiantes")
    students = get_students()
    student_count = count_students()
    st.write(f"Cantidad total de estudiantes: {student_count}")
    for student in students:
        st.write(f"ID: {student['id']}, Nombre: {student['name']}, Edad: {student['age']}")

elif choice == "Agregar":
    st.subheader("Agregar Estudiante")
    name = st.text_input("Nombre")
    age = st.number_input("Edad", min_value=1, max_value=100)
    if st.button("Agregar"):
        add_student(name, age)
        st.success("Estudiante agregado exitosamente")

elif choice == "Actualizar":
    st.subheader("Actualizar Estudiante")
    student_id = st.number_input("ID del estudiante", min_value=1)
    name = st.text_input("Nuevo Nombre")
    age = st.number_input("Nueva Edad", min_value=1, max_value=100)
    if st.button("Actualizar"):
        update_student(student_id, name, age)
        st.success("Estudiante actualizado exitosamente")

elif choice == "Eliminar":
    st.subheader("Eliminar Estudiante")
    student_id = st.number_input("ID del estudiante", min_value=1)
    if st.button("Eliminar"):
        delete_student(student_id)
        st.success("Estudiante eliminado exitosamente")
