import streamlit as st
from supabase import create_client, Client

# Configurar Supabase
SUPABASE_URL = "https://xhnskoldrpeslxhbyami.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBlaW9xd3ZseHJndWpvdGN1YXp0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQwMzM0MDUsImV4cCI6MjAzOTYwOTQwNX0.fLmClBVIcVGr_iKYTw79kPJUb12Iem7beooWfesNiXE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Funciones CRUD
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

# Interfaz de usuario con Streamlit y Bootstrap
st.markdown('''
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .navbar {
            background-color: #C70039;
        }
        .sidebar .sidebar-content {
            background-color: #C70039;
            color: white;
        }
        .main {
            padding: 20px;
        }
        .main h1 {
            color: #C70039;
        }
        .btn-primary {
            background-color: #C70039;
            border-color: #C70039;
        }
    </style>
''', unsafe_allow_html=True)

st.title("CRUD con Streamlit y Supabase")

menu = ["Ver", "Agregar", "Actualizar", "Eliminar"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Ver":
    st.subheader("Lista de estudiantes")
    students = get_students()

    # Convertir la lista de estudiantes a un DataFrame para usar con Streamlit
    df = pd.DataFrame(students)

    # Paginación
    items_per_page = 10
    total_students = len(df)
    total_pages = total_students // items_per_page + (1 if total_students % items_per_page > 0 else 0)

    page = st.number_input('Página', min_value=1, max_value=total_pages, step=1)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    df_page = df.iloc[start_idx:end_idx]

    st.table(df_page)

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
