import streamlit as st
import pandas as pd
from supabase import create_client, Client
import streamlit as st

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

st.image("log_ic-removebg-preview.png", width=200)
st.title("CRUD Python - Instituto Continental IDL3")
# Aplicar CSS para personalizar el menú
st.markdown(
    """
    <style>
    .css-1d391kg {
        background-color: red;  /* Cambia el color de fondo del menú de la barra lateral */
    }
    .css-1d391kg .css-1wa3eu0 {
        color: white;  /* Cambia el color del texto del menú */
    }
    .css-1d391kg .css-1n7v3b2 {
        color: white;  /* Cambia el color del texto del menú cuando está seleccionado */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Definir el menú
menu = ["Ver", "Agregar", "Actualizar", "Eliminar"]
choice = st.sidebar.selectbox("Menú", menu)

# Tu código para manejar la elección del menú aquí
if choice == "Ver":
    st.subheader("Ver")
    # Tu código para la opción "Ver"
elif choice == "Agregar":
    st.subheader("Agregar")
    # Tu código para la opción "Agregar"
elif choice == "Actualizar":
    st.subheader("Actualizar")
    # Tu código para la opción "Actualizar"
elif choice == "Eliminar":
    st.subheader("Eliminar")
    # Tu código para la opción "Eliminar"

def paginate_dataframe(df, page_size):
    """Divide el DataFrame en páginas de un tamaño dado."""
    num_pages = (len(df) + page_size - 1) // page_size
    for page in range(num_pages):
        start_idx = page * page_size
        end_idx = min((page + 1) * page_size, len(df))
        yield df.iloc[start_idx:end_idx]

if choice == "Ver":
    st.subheader("Lista de estudiantes")
    
    # Obtener los datos de los estudiantes
    students = get_students()
    student_count = count_students()
    
    # Mostrar la cantidad total de estudiantes
    st.write(f"Cantidad total de estudiantes: {student_count}")
    
    # Convertir la lista de estudiantes en un DataFrame de pandas
    df_students = pd.DataFrame(students)
    
    # Tamaño de página
    page_size = 5
    
    # Crear la numeración personalizada
    df_students['#'] = df_students.index + 1
    
    # Calcular número de páginas
    num_pages = (len(df_students) + page_size - 1) // page_size
    
    # Selección de página
    page_number = st.number_input("Selecciona la página:", min_value=1, max_value=num_pages, value=1)
    
    # Mostrar la página seleccionada
    for page_df in paginate_dataframe(df_students, page_size):
        if page_number == (df_students.index[page_df.index[-1]] // page_size + 1):
            st.dataframe(page_df[['#', 'id', 'name', 'age']])
            st.write(f"Página {page_number} de {num_pages}")

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
