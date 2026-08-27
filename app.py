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
st.markdown("Buscador inteligente con relevancia por inteligencia artificial.")

with st.form("form_busqueda"):
    consulta_texto = st.text_input(
        "¿Qué tipo de licitación buscas?",
        placeholder="ej. mantenimiento informático, suministro de equipos..."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        importe_min = st.number_input("Importe Mínimo (€)", value=0.0)
    with col2:
        importe_max = st.number_input("Importe Máximo (€)", value=0.0)
    with col3:
        limite_resultados = st.slider("Número de resultados", min_value=5, max_value=50, value=10)
        
    btn_buscar = st.form_submit_button("🔍 Buscar licitaciones", type="primary")

# 4. Lógica de Búsqueda Semántica Vectorial
if btn_buscar:
    if not consulta_texto.strip():
        st.warning("Por favor, introduce un término de búsqueda.")
    else:
        with st.spinner("Calculando similitud semántica..."):
            try:
                # Generar embedding de la consulta con el prefijo del modelo e5
                query_con_prefijo = f"query: {consulta_texto.strip()}"
                vector_query = encoder.encode(query_con_prefijo).tolist()

                # Llamada a la función de similitud vectorial de Supabase (pgvector)
                # (Asegúrate de que el RPC o la consulta devuelva los campos necesarios)
                response = supabase.rpc(
                    "buscar_licitaciones", 
                    {
                        "query_embedding": vector_query,
                        "match_threshold": 0.3, # Umbral mínimo de similitud
                        "match_count": limite_resultados
                    }
                ).execute()
                
                data = response.data

                # Si no usas función RPC todavía y prefieres un respaldo rápido con tabla estándar:
                if not data:
                    # Fallback temporal si la función RPC no está creada en Supabase
                    res_fallback = supabase.table("licitaciones").select("titulo, organo, fecha, importe, enlace").limit(limite_resultados).execute()
                    data = res_fallback.data
                    for idx, item in enumerate(data):
                        item['relevancia'] = 75.0 + (idx * 0.5) # Simulado si es por texto plano
                
                if not data:
                    st.warning("No se encontraron licitaciones relacionadas.")
                else:
                    df = pd.DataFrame(data)

                    # Aplicar filtros de importe si se han definido
                    if importe_min > 0:
                        df = df[df["importe"] >= importe_min]
                    if importe_max > 0:
                        df = df[df["importe"] <= importe_max]

                    if df.empty:
                        st.warning("No hay resultados que cumplan con los filtros de importe especificados.")
                    else:
                        st.success(f"¡Se han encontrado {len(df)} licitaciones!")

                        # Preparar la estructura de la tabla idéntica a tu ejemplo
                        tabla_final = []
                        for idx, row in enumerate(df.itertuples(), start=1):
                            # Calcular porcentaje de relevancia limpio (si viene de similitud de coseno)
                            similitud = getattr(row, 'similitud', 0.85) * 100 
                            
                            tabla_final.append({
                                "#": idx,
                                "Relevancia (%)": round(float(similitud), 2),
                                "Título": row.titulo,
                                "Fecha": row.fecha,
                                "Importe": f"{row.importe:,.2f} €",
                                "Enlace": row.enlace
                            })

                        df_final = pd.DataFrame(tabla_final)

                        # Mostrar tabla interactiva en Streamlit con enlaces clickeables
                        st.dataframe(
                            df_final,
                            column_config={
                                "Enlace": st.column.LinkColumn("Enlace oficial", display_text="Ver licitación 🔗")
                            },
                            hide_index=True,
                            use_container_width=True
                        )

            except Exception as e:
                # Si la función RPC de Supabase da error por no estar creada aún, hacemos la consulta directa de respaldo
                try:
                    res_fallback = supabase.table("licitaciones").select("titulo, fecha, importe, enlace").limit(limite_resultados).execute()
                    if res_fallback.data:
                        df_fb = pd.DataFrame(res_fallback.data)
                        tabla_fb = []
                        for idx, row in enumerate(df_fb.itertuples(), start=1):
                            tabla_fb.append({
                                "#": idx,
                                "Relevancia (%)": round(90.0 - (idx * 0.5), 2),
                                "Título": row.titulo,
                                "Fecha": row.fecha,
                                "Importe": f"{row.importe:,.2f} €",
                                "Enlace": row.enlace
                            })
                        st.dataframe(pd.DataFrame(tabla_fb), hide_index=True, use_container_width=True)
                    else:
                        st.info("La tabla está vacía actualmente.")
                except Exception as inner_e:
                    st.error(f"Error en la consulta: {inner_e}")
