import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# Desactivar traductor automático del navegador
st.markdown(
    """
    <head>
        <meta name="google" content="notranslate">
    </head>
    """,
    unsafe_allow_html=True
)

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Buscador Semántico de Licitaciones", 
    page_icon="🔍", 
    layout="wide"
)

# 1. Configuración de Credenciales
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("⚠️ Faltan las credenciales de Supabase en los Secrets de Streamlit.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Cargar modelo de IA
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')

with st.spinner("Cargando modelo de IA..."):
    encoder = cargar_modelo()

# 3. Interfaz Visual
st.title("🔍 Buscador Semántico de Licitaciones (PLACSP)")
st.markdown("Buscador conectado a Supabase con tus datos reales.")

with st.form("form_busqueda"):
    consulta_texto = st.text_input(
        "¿Qué tipo de licitación buscas?",
        placeholder="ej. obras, servicios, suministros, mantenimiento..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        importe_min = st.number_input("Importe Mínimo (€)", value=0.0)
    with col2:
        importe_max = st.number_input("Importe Máximo (€)", value=0.0)
        
    btn_buscar = st.form_submit_button("🔍 Buscar licitaciones", type="primary")

# 4. Lógica de Búsqueda
if btn_buscar:
    with st.spinner("Buscando en la base de datos..."):
        try:
            # Traer los datos de Supabase (por ahora traemos los registros para filtrarlos e imprimirlos)
            response = supabase.table("licitaciones").select("titulo, organo, fecha, importe, enlace, texto_completo").execute()
            data = response.data

            if not data:
                st.info("No hay licitaciones en la base de datos todavía. Vuelve a ejecutar la carga en Colab.")
            else:
                df = pd.DataFrame(data)
                
                # Filtrar por texto si el usuario ha escrito algo
                if consulta_texto.strip():
                    termino = consulta_texto.strip().lower()
                    # Búsqueda sencilla por coincidencia en el título u órgano
                    df = df[df['texto_completo'].str.lower().str.contains(termino, na=False)]

                # Aplicar filtros de importe
                if importe_min > 0:
                    df = df[df["importe"] >= importe_min]
                if importe_max > 0:
                    df = df[df["importe"] <= importe_max]

                if df.empty:
                    st.warning("No se encontraron resultados con los filtros o términos indicados.")
                else:
                    st.success(f"¡Se han encontrado {len(df)} licitaciones!")
                    
                    # Mostrar resultados ordenados con el diseño limpio
                    for _, row in df.iterrows():
                        with st.container(border=True):
                            st.subheader(row["titulo"])
                            st.write(f"🏢 **Órgano:** {row['organo']} | 📅 **Fecha:** {row['fecha']} | 💰 **Importe:** {row['importe']:,.2f} €")
                            st.markdown(f"[Ver licitación oficial en la PLACSP 🔗]({row['enlace']})")
        
        except Exception as e:
            st.error(f"Error al realizar la consulta: {e}")
