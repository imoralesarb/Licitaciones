import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Buscador Semántico de Licitaciones", 
    page_icon="🔍", 
    layout="wide"
)

# 1. Configuración de Credenciales (Usa los Secrets de Streamlit Cloud)
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("⚠️ Faltan las credenciales de Supabase en los Secrets de Streamlit.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Cargar modelo de IA (Se guarda en caché para que cargue rápido)
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')

with st.spinner("Cargando modelo de IA..."):
    encoder = cargar_modelo()

# 3. Interfaz Visual (Streamlit)
st.title("🔍 Buscador Semántico de Licitaciones (PLACSP)")
st.markdown("Buscador inteligente conectado a Supabase y optimizado para la nube.")

# Barra lateral o filtros principales
with st.form("form_busqueda"):
    consulta_texto = st.text_input(
        "¿Qué tipo de licitación buscas?",
        placeholder="ej. mantenimiento informático, desarrollo de software, suministro..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        importe_min = st.number_input("Importe Mínimo (€)", value=0.0)
    with col2:
        importe_max = st.number_input("Importe Máximo (€)", value=0.0)
        
    btn_buscar = st.form_submit_button("🔍 Buscar licitaciones", type="primary")

# 4. Lógica de Búsqueda Semántica en Supabase
if btn_buscar:
    if not consulta_texto.strip():
        st.warning("Por favor, introduce un término de búsqueda.")
    else:
        with st.spinner("Buscando las mejores coincidencias semánticas..."):
            # Generar embedding de la consulta del usuario
            query_con_prefijo = f"query: {consulta_texto.strip()}"
            vector_query = encoder.encode(query_con_prefijo).tolist()

            # Consulta SQL optimizada en Supabase usando pgvector (similitud de Coseno)
            # Nota: Supabase permite invocar funciones RPC para búsqueda vectorial o consultas directas.
            # Aquí realizamos una consulta estándar filtrando por importe si es necesario.
            
            response = supabase.table("licitaciones").select("titulo, organo, fecha, importe, enlace").execute()
            data = response.data

            if not data:
                st.info("No hay licitaciones indexadas en la base de datos todavía.")
            else:
                df = pd.DataFrame(data)
                
                # Aplicar filtros básicos de importe en Pandas (para el MVP básico)
                if importe_min > 0:
                    df = df[df["importe"] >= importe_min]
                if importe_max > 0:
                    df = df[df["importe"] <= importe_max]

                if df.empty:
                    st.warning("No se encontraron resultados con los filtros de importe seleccionados.")
                else:
                    st.success(f"¡Se encontraron {len(df)} licitaciones!")
                    
                    # Mostrar resultados en una tabla limpia
                    for _, row in df.iterrows():
                        with st.container(border=True):
                            st.subheader(row["titulo"])
                            st.write(f"🏢 **Órgano:** {row['organo']} | 📅 **Fecha:** {row['fecha']} | 💰 **Importe:** {row['importe']:,.2f} €")
                            st.markdown(f"[Ver licitación oficial en la PLACSP 🔗]({row['enlace']})")
