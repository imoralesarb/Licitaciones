import os
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client
from datetime import date

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
    page_title="Buscador inteligente de Licitaciones", 
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

# 3. Descargar datos de Supabase incluyendo los nuevos campos para filtros
@st.cache_data(ttl=600) # Se actualiza cada 10 minutos
def obtener_datos_supabase():
    response = supabase.table("licitaciones").select("titulo, organo, fecha, importe, enlace, lugar_ejecucion, fecha_fin, texto_completo, embedding").execute()
    return response.data

# Mapa territorial ampliado con opciones generales y específicas (islas/provincias)
MAPA_TERRITORIAL = {
    "Todas las CCAA / Ubicaciones": [],
    # --- CANARIAS Y SUS ISLAS/PROVINCIAS ---
    "Canarias (General)": ["Canarias", "Tenerife", "Gran Canaria", "Lanzarote", "Fuerteventura", "La Palma", "La Gomera", "El Hierro", "Las Palmas", "Santa Cruz de Tenerife"],
    "   ↳ La Gomera": ["La Gomera"],
    "   ↳ Tenerife": ["Tenerife", "Santa Cruz de Tenerife"],
    "   ↳ Gran Canaria": ["Gran Canaria", "Las Palmas"],
    "   ↳ Lanzarote": ["Lanzarote"],
    "   ↳ Fuerteventura": ["Fuerteventura"],
    "   ↳ La Palma": ["La Palma"],
    "   ↳ El Hierro": ["El Hierro"],
    # --- RESTO DE CCAA Y PROVINCIAS ---
    "Andalucía (General)": ["Andalucía", "Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"],
    "   ↳ Almería": ["Almería"],
    "   ↳ Cádiz": ["Cádiz"],
    "   ↳ Córdoba": ["Córdoba"],
    "   ↳ Granada": ["Granada"],
    "   ↳ Huelva": ["Huelva"],
    "   ↳ Jaén": ["Jaén"],
    "   ↳ Málaga": ["Málaga"],
    "   ↳ Sevilla": ["Sevilla"],
    "Aragón": ["Aragón", "Huesca", "Teruel", "Zaragoza"],
    "Asturias (Principado de)": ["Asturias", "Oviedo", "Gijón"],
    "Illes Balears / Islas Baleares": ["Baleares", "Balears", "Mallorca", "Menorca", "Ibiza", "Formentera", "Palma"],
    "Cantabria": ["Cantabria", "Santander"],
    "Castilla-La Mancha": ["Castilla-La Mancha", "Albacete", "Ciudad Real", "Cuenca", "Guadalajara", "Toledo"],
    "Castilla y León": ["Castilla y León", "Ávila", "Burgos", "León", "Palencia", "Salamanca", "Segovia", "Soria", "Valladolid", "Zamora"],
    "Cataluña": ["Cataluña", "Catalunya", "Barcelona", "Gerona", "Girona", "Lérida", "Lleida", "Tarragona"],
    "Comunitat Valenciana": ["Valenciana", "Valencia", "Alicante", "Castellón"],
    "Extremadura": ["Extremadura", "Badajoz", "Cáceres"],
    "Galicia": ["Galicia", "Coruña", "A Coruña", "Lugo", "Ourense", "Orense", "Pontevedra", "Vigo"],
    "Madrid (Comunidad de)": ["Madrid"],
    "Murcia (Región de)": ["Murcia"],
    "Navarra (Comunidad Foral de)": ["Navarra", "Pamplona"],
    "País Vasco": ["País Vasco", "Euskadi", "Álava", "Araba", "Guipúzcoa", "Gipuzkoa", "Vizcaya", "Bizkaia", "Bilbao", "San Sebastián", "Vitoria"],
    "La Rioja": ["La Rioja", "Logroño"],
    "Ceuta": ["Ceuta"],
    "Melilla": ["Melilla"]
}

# 4. Interfaz Visual y Gestión de Estado
st.title("🔍 Buscador inteligente de Licitaciones (PLACSP)")

def limpiar_campos():
    st.session_state.consulta_texto = ""
    st.session_state.filtro_ccaa = "Todas las CCAA / Ubicaciones"
    st.session_state.filtro_fecha_cierre = ""
    st.session_state.importe_min = 0.0
    st.session_state.importe_max = 0.0
    st.session_state.limite_resultados = 10
    st.session_state.mostrar_todos = False
    st.session_state.usar_filtro_fechas = False

# Buscador principal
consulta_texto = st.text_input(
    "¿Qué tipo de licitación buscas? (Opcional)",
    placeholder="ej. mantenimiento informático, suministro de vehículos, obras...",
    key="consulta_texto"
)

# Panel de filtros integrados en la misma vista
st.markdown("### ⚙️ Filtros avanzados")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    importe_min = st.number_input("Importe Mínimo (€)", value=0.0, key="importe_min")
with col2:
    importe_max = st.number_input("Importe Máximo (€)", value=0.0, key="importe_max")
with col3:
    # Desplegable con opciones generales y específicas (ej. Canarias o directamente La Gomera)
    lista_ccaa = list(MAPA_TERRITORIAL.keys())
    filtro_ccaa = st.selectbox("📍 Lugar (CCAA / Isla)", lista_ccaa, key="filtro_ccaa")
with col4:
    filtro_fecha_cierre = st.text_input("⏳ Fecha fin (texto/parcial)", placeholder="ej. 2026-09", key="filtro_fecha_cierre")
with col5:
    limite_resultados = st.slider("Resultados", min_value=1, max_value=500, value=10, key="limite_resultados")

# Opción para mostrar todos los resultados posibles, control de fechas y Botón de Limpiar
col_chk1, col_chk2, col_btn = st.columns([1, 2, 1])
with col_chk1:
    mostrar_todos = st.checkbox("Mostrar TODOS", key="mostrar_todos")

with col_chk2:
    usar_filtro_fechas = st.checkbox("📅 Activar rango de fechas de publicación", key="usar_filtro_fechas")

with col_btn:
    st.write("") 
    st.button("🔄 Limpiar Filtros", on_click=limpiar_campos, type="secondary")

if usar_filtro_fechas:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_inicio = st.date_input("Desde", value=date(2026, 1, 1))
    with col_f2:
        f_fin = st.date_input("Hasta", value=date(2026, 12, 31))

btn_buscar = st.button("🔍 Buscar licitaciones", type="primary")

# 5. Lógica de Búsqueda y Filtrado
if btn_buscar:
    with st.spinner("Procesando licitaciones..."):
        data = obtener_datos_supabase()

        if not data:
            st.info("No hay licitaciones en la base de datos de Supabase.")
        else:
            df = pd.DataFrame(data)
            
            if consulta_texto.strip():
                if "embedding" not in df.columns or df["embedding"].isnull().all():
                    st.error("⚠️ Los registros en Supabase no contienen vectores (embeddings). Vuelve a realizar la carga.")
                else:
                    query_con_prefijo = f"query: {consulta_texto.strip()}"
                    vector_query = encoder.encode(query_con_prefijo, convert_to_tensor=True)

                    vectores_tensor = encoder.encode(df["texto_completo"].tolist(), convert_to_tensor=True)
                    cos_scores = util.cos_sim(vector_query, vectores_tensor)[0]
                    
                    df["relevancia"] = (cos_scores.cpu().numpy() * 100).round(2)
                    df = df.sort_values(by="relevancia", ascending=False)
            else:
                df["relevancia"] = 100.0 
                if "fecha" in df.columns:
                    df = df.sort_values(by="fecha", ascending=False)

            # --- APLICAR FILTROS ---
            if importe_min > 0:
                df = df[df["importe"] >= importe_min]
            if importe_max > 0:
                df = df[df["importe"] <= importe_max]
            
            # Filtro inteligente por CCAA o isla específica seleccionada
            if filtro_ccaa != "Todas las CCAA / Ubicaciones":
                palabras_clave = MAPA_TERRITORIAL.get(filtro_ccaa, [filtro_ccaa])
                patron_regex = '|'.join([r'\b' + p + r'\b' for p in palabras_clave])
                df = df[df["lugar_ejecucion"].str.contains(patron_regex, case=False, na=False)]
            
            if filtro_fecha_cierre.strip():
                df = df[df["fecha_fin"].str.contains(filtro_fecha_cierre.strip(), case=False, na=False)]

            if usar_filtro_fechas:
                def filtrar_fecha(f_str):
                    if not f_str:
                        return False
                    try:
                        obj_f = date.fromisoformat(f_str[:10])
                        return f_inicio <= obj_f <= f_fin
                    except ValueError:
                        return False
                df = df[df["fecha"].apply(filtrar_fecha)]

            if not mostrar_todos:
                df = df.head(limite_resultados)

            if df.empty:
                st.warning("No se encontraron resultados que cumplan con los filtros indicados.")
            else:
                st.success(f"¡Se han encontrado {len(df)} licitaciones relevantes!")

                tabla_final = []
                for idx, row in enumerate(df.itertuples(), start=1):
                    tabla_final.append({
                        "#": idx,
                        "Relevancia (%)": getattr(row, "relevancia", 100.0),
                        "Título": row.titulo,
                        "Órgano": row.organo,
                        "Lugar": getattr(row, "lugar_ejecucion", "No especificado"),
                        "Cierre": getattr(row, "fecha_fin", "No especificada"),
                        "Fecha Pub.": row.fecha,
                        "Importe": f"{row.importe:,.2f} €",
                        "Enlace": row.enlace
                    })

                df_final = pd.DataFrame(tabla_final)

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
