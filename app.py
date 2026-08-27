import os
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
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

# 2. Cargar modelo de IA en caché
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')

with st.spinner("Cargando modelo de IA..."):
    encoder = cargar_modelo()

# 3. Descargar datos de Supabase en caché para búsqueda ultra rápida y precisa
@st.cache_data(ttl=600) # Se actualiza cada 10 minutos
def obtener_datos_supabase():
    response = supabase.table("licitaciones").select("titulo, organo, fecha, importe, enlace, texto_completo, embedding").execute()
    return response.data

# 4. Interfaz Visual
st.title("🔍 Buscador Semántico de Licitaciones (PLACSP)")
st.markdown("Buscador inteligente optimizado con IA y filtrado vectorial local.")

# Usamos componentes fuera de un form estricto para permitir múltiples búsquedas fluidas
consulta_texto = st.text_input(
    "¿Qué tipo de licitación buscas?",
    placeholder="ej. mantenimiento informático, suministro de vehículos, obras..."
)

col1, col2, col3 = st.columns(3)
with col1:
    importe_min = st.number_input("Importe Mínimo (€)", value=0.0)
with col2:
    importe_max = st.number_input("Importe Máximo (€)", value=0.0)
with col3:
    limite_resultados = st.slider("Número de resultados", min_value=5, max_value=50, value=10)

btn_buscar = st.button("🔍 Buscar licitaciones", type="primary")

# 5. Lógica de Búsqueda Semántica Real
if btn_buscar:
    if not consulta_texto.strip():
        st.warning("Por favor, introduce un término de búsqueda.")
    else:
        with st.spinner("Analizando similitud semántica con IA..."):
            data = obtener_datos_supabase()

            if not data:
                st.info("No hay licitaciones en la base de datos de Supabase.")
            else:
                df = pd.DataFrame(data)
                
                # Verificar si existen embeddings guardados
                if "embedding" not in df.columns or df["embedding"].isnull().all():
                    st.error("⚠️ Los registros en Supabase no contienen vectores (embeddings). Vuelve a realizar la carga.")
                else:
                    # Generar embedding de la consulta del usuario (con prefijo del modelo e5)
                    query_con_prefijo = f"query: {consulta_texto.strip()}"
                    vector_query = encoder.encode(query_con_prefijo, convert_to_tensor=True)

                    # Extraer todos los vectores de la base de datos
                    vectores_db = np.array(df["embedding"].tolist())
                    vectores_tensor = encoder.encode(df["texto_completo"].tolist(), convert_to_tensor=True)

                    # Calcular similitud de Coseno matemáticamente de forma precisa
                    cos_scores = util.cos_sim(vector_query, vectores_tensor)[0]
                    
                    # Añadir la puntuación de relevancia al DataFrame (convertida a porcentaje de 0 a 100)
                    df["relevancia"] = (cos_scores.cpu().numpy() * 100).round(2)

                    # Ordenar de mayor a menor relevancia
                    df = df.sort_values(by="relevancia", ascending=False)

                    # Aplicar filtros de importe si se han definido
                    if importe_min > 0:
                        df = df[df["importe"] >= importe_min]
                    if importe_max > 0:
                        df = df[df["importe"] <= importe_max]

                    # Limitar al número de resultados seleccionados
                    df = df.head(limite_resultados)

                    if df.empty:
                        st.warning("No se encontraron resultados que cumplan con los filtros de importe indicados.")
                    else:
                        st.success(f"¡Se han encontrado {len(df)} licitaciones relevantes!")

                        # Preparar la estructura de la tabla exacta que pediste
                        tabla_final = []
                        for idx, row in enumerate(df.itertuples(), start=1):
                            tabla_final.append({
                                "#": idx,
                                "Relevancia (%)": row.relevancia,
                                "Título": row.titulo,
                                "Fecha": row.fecha,
                                "Importe": f"{row.importe:,.2f} €",
                                "Enlace": row.enlace
                            })

                        df_final = pd.DataFrame(tabla_final)

                        # Mostrar tabla interactiva con enlaces 100% clickeables corregidos
                        st.dataframe(
                            df_final,
                            column_config={
                                "Enlace": st.column_config.LinkColumn(
                                    "Enlace oficial", 
                                    display_text="Ver licitación 🔗"
                                )
                            },
                            hide_index=True,
                            use_container_width=True
                        )
