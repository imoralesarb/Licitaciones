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

# --- ESTILOS CSS PERSONALIZADOS PARA LOS BOTONES ---
st.markdown(
    """
    <style>
        div.stButton > button:first-child {
            background-color: #0066cc;
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            border: none;
            width: 100%;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #0052a3;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
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
@st.cache_data(ttl=600)
def obtener_datos_supabase():
    response = supabase.table("licitaciones").select("titulo, organo, fecha, importe, enlace, lugar_ejecucion, fecha_fin, texto_completo, embedding, cpv").execute()
    return response.data

# MAPA TERRITORIAL COMPLETO DE ESPAÑA
MAPA_TERRITORIAL = {
    "🌐 Todas las CCAA / Ubicaciones": [],
    
    # --- ANDALUCÍA ---
    "📍 Andalucía (General)": ["Andalucía", "Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"],
    "    ↳ Almería": ["Almería"],
    "    ↳ Cádiz": ["Cádiz"],
    "    ↳ Córdoba": ["Córdoba"],
    "    ↳ Granada": ["Granada"],
    "    ↳ Huelva": ["Huelva"],
    "    ↳ Jaén": ["Jaén"],
    "    ↳ Málaga": ["Málaga"],
    "    ↳ Sevilla": ["Sevilla"],

    # --- ARAGÓN ---
    "📍 Aragón (General)": ["Aragón", "Huesca", "Teruel", "Zaragoza"],
    "    ↳ Huesca": ["Huesca"],
    "    ↳ Teruel": ["Teruel"],
    "    ↳ Zaragoza": ["Zaragoza"],

    # --- ASTURIAS ---
    "📍 Asturias (Principado de)": ["Asturias", "Oviedo", "Gijón", "Avilés"],

    # --- BALEARES ---
    "📍 Illes Balears / Islas Baleares (General)": ["Baleares", "Balears", "Mallorca", "Menorca", "Ibiza", "Formentera", "Palma"],
    "    ↳ Mallorca / Palma": ["Mallorca", "Palma"],
    "    ↳ Menorca": ["Menorca"],
    "    ↳ Ibiza y Formentera": ["Ibiza", "Formentera"],

    # --- CANARIAS ---
    "📍 Canarias (General)": ["Canarias", "Tenerife", "Gran Canaria", "Lanzarote", "Fuerteventura", "La Palma", "La Gomera", "El Hierro", "Las Palmas", "Santa Cruz de Tenerife"],
    "    ↳ Tenerife": ["Tenerife", "Santa Cruz de Tenerife"],
    "    ↳ Gran Canaria": ["Gran Canaria", "Las Palmas"],
    "    ↳ Lanzarote": ["Lanzarote"],
    "    ↳ Fuerteventura": ["Fuerteventura"],
    "    ↳ La Palma": ["La Palma"],
    "    ↳ La Gomera": ["La Gomera"],
    "    ↳ El Hierro": ["El Hierro"],

    # --- CANTABRIA ---
    "📍 Cantabria": ["Cantabria", "Santander"],

    # --- CASTILLA-LA MANCHA ---
    "📍 Castilla-La Mancha (General)": ["Castilla-La Mancha", "Albacete", "Ciudad Real", "Cuenca", "Guadalajara", "Toledo"],
    "    ↳ Albacete": ["Albacete"],
    "    ↳ Ciudad Real": ["Ciudad Real"],
    "    ↳ Cuenca": ["Cuenca"],
    "    ↳ Guadalajara": ["Guadalajara"],
    "    ↳ Toledo": ["Toledo"],

    # --- CASTILLA Y LEÓN ---
    "📍 Castilla y León (General)": ["Castilla y León", "Ávila", "Burgos", "León", "Palencia", "Salamanca", "Segovia", "Soria", "Valladolid", "Zamora"],
    "    ↳ Ávila": ["Ávila"],
    "    ↳ Burgos": ["Burgos"],
    "    ↳ León": ["León"],
    "    ↳ Palencia": ["Palencia"],
    "    ↳ Salamanca": ["Salamanca"],
    "    ↳ Segovia": ["Segovia"],
    "    ↳ Soria": ["Soria"],
    "    ↳ Valladolid": ["Valladolid"],
    "    ↳ Zamora": ["Zamora"],

    # --- CATALUÑA ---
    "📍 Cataluña / Catalunya (General)": ["Cataluña", "Catalunya", "Barcelona", "Gerona", "Girona", "Lérida", "Lleida", "Tarragona"],
    "    ↳ Barcelona": ["Barcelona"],
    "    ↳ Girona / Gerona": ["Gerona", "Girona"],
    "    ↳ Lleida / Lérida": ["Lérida", "Lleida"],
    "    ↳ Tarragona": ["Tarragona"],

    # --- COMUNITAT VALENCIANA ---
    "📍 Comunitat Valenciana (General)": ["Valenciana", "Valencia", "Alicante", "Castellón"],
    "    ↳ Alicante / Alacant": ["Alicante"],
    "    ↳ Castellón / Castelló": ["Castellón"],
    "    ↳ Valencia / València": ["Valencia"],

    # --- EXTREMADURA ---
    "📍 Extremadura (General)": ["Extremadura", "Badajoz", "Cáceres"],
    "    ↳ Badajoz": ["Badajoz"],
    "    ↳ Cáceres": ["Cáceres"],

    # --- GALICIA ---
    "📍 Galicia (General)": ["Galicia", "Coruña", "A Coruña", "Lugo", "Ourense", "Orense", "Pontevedra", "Vigo"],
    "    ↳ A Coruña / Coruña": ["Coruña", "A Coruña"],
    "    ↳ Lugo": ["Lugo"],
    "    ↳ Ourense / Orense": ["Ourense", "Orense"],
    "    ↳ Pontevedra / Vigo": ["Pontevedra", "Vigo"],

    # --- MADRID ---
    "📍 Madrid (Comunidad de)": ["Madrid"],

    # --- MURCIA ---
    "📍 Murcia (Región de)": ["Murcia"],

    # --- NAVARRA ---
    "📍 Navarra (Comunidad Foral de)": ["Navarra", "Pamplona"],

    # --- PAÍS VASCO ---
    "📍 País Vasco / Euskadi (General)": ["País Vasco", "Euskadi", "Álava", "Araba", "Guipúzcoa", "Gipuzkoa", "Vizcaya", "Bizkaia", "Bilbao", "San Sebastián", "Vitoria"],
    "    ↳ Álava / Araba": ["Álava", "Araba", "Vitoria"],
    "    ↳ Guipúzcoa / Gipuzkoa": ["Guipúzcoa", "Gipuzkoa", "San Sebastián"],
    "    ↳ Vizcaya / Bizkaia": ["Vizcaya", "Bizkaia", "Bilbao"],

    # --- LA RIOJA ---
    "📍 La Rioja": ["La Rioja", "Logroño"],

    # --- CIUDADES AUTÓNOMAS ---
    "📍 Ceuta": ["Ceuta"],
    "📍 Melilla": ["Melilla"]
}

# MAPA DE SECTORES CPV OFICIALES
SECTORES_CPV = {
    "🌐 Todos los sectores CPV": [],
    "Agricultura, alimentación y materias primas": ["03", "09", "14", "15", "16"],
    "Textil, industria, maquinaria y bienes de consumo": ["18", "19", "22", "24", "30", "31", "32", "33", "34", "35", "37", "38", "39"],
    "Construcción, agua y energía": ["41", "42", "43", "44", "45", "48"],
    "Servicios generales a empresas y mantenimiento": ["50", "51", "55"],
    "Transporte, correos y telecomunicaciones": ["60", "63", "64", "65"],
    "Servicios financieros, inmobiliarios y profesionales": ["66", "70", "71", "72", "73", "75", "76", "77", "79"],
    "Educación, sanidad, medio ambiente y servicios sociales": ["80", "85", "90", "92", "98"]
}

# 4. Interfaz Visual y Gestión de Estado
st.title("🔍 Buscador inteligente de Licitaciones")

def limpiar_campos():
    st.session_state.consulta_texto = ""
    st.session_state.filtro_ccaa = "🌐 Todas las CCAA / Ubicaciones"
    st.session_state.filtro_lugar_libre = ""
    st.session_state.filtro_cpv_sector = "🌐 Todos los sectores CPV"
    st.session_state.importe_min = 0.0
    st.session_state.importe_max = 0.0
    st.session_state.limite_resultados = 10
    st.session_state.mostrar_todos = False
    st.session_state.usar_filtro_fechas = False
    st.session_state.usar_filtro_cierre = False

# Buscador principal
consulta_texto = st.text_input(
    "¿Qué tipo de licitación buscas?",
    placeholder="ej. mantenimiento informático, suministro de vehículos, obras...",
    key="consulta_texto"
)

# Panel de filtros integrados en la misma vista
st.markdown("### ⚙️ Filtros avanzados")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    importe_min = st.number_input("Importe Mínimo (€)", value=0.0, key="importe_min")
with col2:
    importe_max = st.number_input("Importe Máximo (€)", value=0.0, key="importe_max")
with col3:
    lista_ccaa = list(MAPA_TERRITORIAL.keys())
    filtro_ccaa = st.selectbox("📍 Ubicación (Desplegable)", lista_ccaa, key="filtro_ccaa")
with col4:
    filtro_lugar_libre = st.text_input("📍 Lugar (Libre)", placeholder="ej. San Sebastián", key="filtro_lugar_libre")
with col5:
    lista_sectores = list(SECTORES_CPV.keys())
    filtro_cpv_sector = st.selectbox("📦 Sector CPV", lista_sectores, key="filtro_cpv_sector")
with col6:
    limite_resultados = st.slider("Resultados", min_value=1, max_value=500, value=10, key="limite_resultados")

# Controles secundarios y calendarios de fechas
col_chk1, col_chk2, col_chk3 = st.columns([1, 2, 2])
with col_chk1:
    mostrar_todos = st.checkbox("Mostrar TODOS", key="mostrar_todos")
with col_chk2:
    usar_filtro_fechas = st.checkbox("📅 Rango fecha publicación en plataforma", key="usar_filtro_fechas")
with col_chk3:
    usar_filtro_cierre = st.checkbox("⏳ Fecha fin de presentación de oferta", key="usar_filtro_cierre")

# Desplegables de calendarios si están activos
if usar_filtro_fechas:
    st.markdown("##### Rango de fecha de publicación")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_inicio = st.date_input("Desde", value=date(2026, 1, 1))
    with col_f2:
        f_fin = st.date_input("Hasta", value=date(2026, 12, 31))

if usar_filtro_cierre:
    st.markdown("##### Fecha fin de presentación de oferta (Muestra las que acaban en este día o después)")
    col_c1, _ = st.columns([1, 1])
    with col_c1:
        fecha_cierre_tope = st.date_input("Fecha tope mínima de fin de presentación", value=date(2026, 3, 1))

st.write("") 

# --- BOTONES DE ACCIÓN PRINCIPAL EN LA MISMA LÍNEA ---
col_btn_buscar, col_btn_limpiar, col_vacio = st.columns([2, 2, 4])

with col_btn_buscar:
    btn_buscar = st.button("🔍 Buscar licitaciones", type="primary", use_container_width=True)

with col_btn_limpiar:
    btn_limpiar = st.button("🔄 Limpiar Filtros", on_click=limpiar_campos, type="secondary", use_container_width=True)


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
            if not df.empty and importe_min > 0:
                df = df[df["importe"] >= importe_min]
            if not df.empty and importe_max > 0:
                df = df[df["importe"] <= importe_max]
            
            # Filtro por desplegable territorial
            if not df.empty and filtro_ccaa != "🌐 Todas las CCAA / Ubicaciones":
                palabras_clave = MAPA_TERRITORIAL.get(filtro_ccaa, [filtro_ccaa])
                patron_regex = '|'.join([r'\b' + p + r'\b' for p in palabras_clave])
                df = df[df["lugar_ejecucion"].str.contains(patron_regex, case=False, na=False)]
            
            # Filtro adicional por texto libre de lugar específico
            if not df.empty and filtro_lugar_libre.strip():
                df = df[df["lugar_ejecucion"].str.contains(filtro_lugar_libre.strip(), case=False, na=False)]

            # Filtro por desplegable de Sector CPV
            if not df.empty and filtro_cpv_sector != "🌐 Todos los sectores CPV":
                prefijos_validos = tuple(SECTORES_CPV[filtro_cpv_sector])
                def coincide_cpv(cpv_str):
                    if not cpv_str or pd.isna(cpv_str) or cpv_str == "No especificado":
                        return False
                    lista_cpv = [c.strip() for c in str(cpv_str).split(",")]
                    return any(c.startswith(prefijos_validos) for c in lista_cpv)
                if "cpv" in df.columns:
                    df = df[df["cpv"].apply(coincide_cpv)]
            
            # Filtro inteligente de fecha fin (mantiene las que expiren en la fecha seleccionada o más adelante)
            if not df.empty and usar_filtro_cierre:
                def filtrar_fecha_fin(f_str):
                    if not f_str:
                        return False
                    try:
                        obj_f = date.fromisoformat(f_str[:10])
                        return obj_f >= fecha_cierre_tope
                    except ValueError:
                        return False
                if "fecha_fin" in df.columns:
                    df = df[df["fecha_fin"].apply(filtrar_fecha_fin)]

            # Filtro de rango de fecha de publicación
            if not df.empty and usar_filtro_fechas:
                def filtrar_fecha_pub(f_str):
                    if not f_str:
                        return False
                    try:
                        obj_f = date.fromisoformat(f_str[:10])
                        return f_inicio <= obj_f <= f_fin
                    except ValueError:
                        return False
                if "fecha" in df.columns:
                    df = df[df["fecha"].apply(filtrar_fecha_pub)]

            if not df.empty and not mostrar_todos:
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
