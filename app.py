from datetime import date
import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

# Desactivar traductor automático del navegador
st.markdown(
    """
    <head>
        <meta name="google" content="notranslate">
    </head>
    """,
    unsafe_allow_html=True,
)

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Buscador inteligente de Licitaciones",
    page_icon="🔍",
    layout="wide",
)

# --- ESTILOS CSS PERSONALIZADOS ---
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
    unsafe_allow_html=True,
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
    return SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")


with st.spinner("Cargando modelo de IA..."):
    encoder = cargar_modelo()


# 3. Descarga auxiliar paginada para el botón de Novedades
@st.cache_data(ttl=600)
def obtener_novedades_supabase():
    todos_los_datos = []
    tamano_lote = 1000
    inicio = 0
    while True:
        response = (
            supabase.table("licitaciones")
            .select(
                "titulo, organo, fecha, importe, enlace, lugar_ejecucion, fecha_fin,"
                " cpv, fuente, es_novedad, es_actualizada"
            )
            .range(inicio, inicio + tamano_lote - 1)
            .execute()
        )
        filas = response.data
        if not filas:
            break
        todos_los_datos.extend(filas)
        if len(filas) < tamano_lote:
            break
        inicio += tamano_lote
    return todos_los_datos


# MAPA TERRITORIAL Y SECTORES
MAPA_TERRITORIAL = {
    "🌐 Todas las CCAA / Ubicaciones": [],
    "📍 Andalucía (General)": [
        "Andalucía",
        "Almería",
        "Cádiz",
        "Córdoba",
        "Granada",
        "Huelva",
        "Jaén",
        "Málaga",
        "Sevilla",
    ],
    "    ↳ Almería": ["Almería"],
    "    ↳ Cádiz": ["Cádiz"],
    "    ↳ Córdoba": ["Córdoba"],
    "    ↳ Granada": ["Granada"],
    "    ↳ Huelva": ["Huelva"],
    "    ↳ Jaén": ["Jaén"],
    "    ↳ Málaga": ["Málaga"],
    "    ↳ Sevilla": ["Sevilla"],
    "📍 Aragón (General)": ["Aragón", "Huesca", "Teruel", "Zaragoza"],
    "    ↳ Huesca": ["Huesca"],
    "    ↳ Teruel": ["Teruel"],
    "    ↳ Zaragoza": ["Zaragoza"],
    "📍 Asturias (Principado de)": ["Asturias", "Oviedo", "Gijón", "Avilés"],
    "📍 Illes Balears / Islas Baleares (General)": [
        "Baleares",
        "Balears",
        "Mallorca",
        "Menorca",
        "Ibiza",
        "Formentera",
        "Palma",
    ],
    "    ↳ Mallorca / Palma": ["Mallorca", "Palma"],
    "    ↳ Menorca": ["Menorca"],
    "    ↳ Ibiza y Formentera": ["Ibiza", "Formentera"],
    "📍 Canarias (General)": [
        "Canarias",
        "Tenerife",
        "Gran Canaria",
        "Lanzarote",
        "Fuerteventura",
        "La Palma",
        "La Gomera",
        "El Hierro",
        "Las Palmas",
        "Santa Cruz de Tenerife",
    ],
    "    ↳ Tenerife": ["Tenerife", "Santa Cruz de Tenerife"],
    "    ↳ Gran Canaria": ["Gran Canaria", "Las Palmas"],
    "    ↳ Lanzarote": ["Lanzarote"],
    "    ↳ Fuerteventura": ["Fuerteventura"],
    "    ↳ La Palma": ["La Palma"],
    "    ↳ La Gomera": ["La Gomera"],
    "    ↳ El Hierro": ["El Hierro"],
    "📍 Cantabria": ["Cantabria", "Santander"],
    "📍 Castilla-La Mancha (General)": [
        "Castilla-La Mancha",
        "Albacete",
        "Ciudad Real",
        "Cuenca",
        "Guadalajara",
        "Toledo",
    ],
    "    ↳ Albacete": ["Albacete"],
    "    ↳ Ciudad Real": ["Ciudad Real"],
    "    ↳ Cuenca": ["Cuenca"],
    "    ↳ Guadalajara": ["Guadalajara"],
    "    ↳ Toledo": ["Toledo"],
    "📍 Castilla y León (General)": [
        "Castilla y León",
        "Ávila",
        "Burgos",
        "León",
        "Palencia",
        "Salamanca",
        "Segovia",
        "Soria",
        "Valladolid",
        "Zamora",
    ],
    "    ↳ Ávila": ["Ávila"],
    "    ↳ Burgos": ["Burgos"],
    "    ↳ León": ["León"],
    "    ↳ Palencia": ["Palencia"],
    "    ↳ Salamanca": ["Salamanca"],
    "    ↳ Segovia": ["Segovia"],
    "    ↳ Soria": ["Soria"],
    "    ↳ Valladolid": ["Valladolid"],
    "    ↳ Zamora": ["Zamora"],
    "📍 Cataluña / Catalunya (General)": [
        "Cataluña",
        "Catalunya",
        "Barcelona",
        "Gerona",
        "Girona",
        "Lérida",
        "Lleida",
        "Tarragona",
    ],
    "    ↳ Barcelona": ["Barcelona"],
    "    ↳ Girona / Gerona": ["Gerona", "Girona"],
    "    ↳ Lleida / Lérida": ["Lérida", "Lleida"],
    "    ↳ Tarragona": ["Tarragona"],
    "📍 Comunitat Valenciana (General)": [
        "Valenciana",
        "Valencia",
        "Alicante",
        "Castellón",
    ],
    "    ↳ Alicante / Alacant": ["Alicante"],
    "    ↳ Castellón / Castelló": ["Castellón"],
    "    ↳ Valencia / València": ["Valencia"],
    "📍 Extremadura (General)": ["Extremadura", "Badajoz", "Cáceres"],
    "    ↳ Badajoz": ["Badajoz"],
    "    ↳ Cáceres": ["Cáceres"],
    "📍 Galicia (General)": [
        "Galicia",
        "Coruña",
        "A Coruña",
        "Lugo",
        "Ourense",
        "Orense",
        "Pontevedra",
        "Vigo",
    ],
    "    ↳ A Coruña / Coruña": ["Coruña", "A Coruña"],
    "    ↳ Lugo": ["Lugo"],
    "    ↳ Ourense / Orense": ["Ourense", "Orense"],
    "    ↳ Pontevedra / Vigo": ["Pontevedra", "Vigo"],
    "📍 Madrid (Comunidad de)": ["Madrid"],
    "📍 Murcia (Región de)": ["Murcia"],
    "📍 Navarra (Comunidad Foral de)": ["Navarra", "Pamplona"],
    "📍 País Vasco / Euskadi (General)": [
        "País Vasco",
        "Euskadi",
        "Álava",
        "Araba",
        "Guipúzcoa",
        "Gipuzkoa",
        "Vizcaya",
        "Bizkaia",
        "Bilbao",
        "San Sebastián",
        "Vitoria",
    ],
    "    ↳ Álava / Araba": ["Álava", "Araba", "Vitoria"],
    "    ↳ Guipúzcoa / Gipuzkoa": ["Guipúzcoa", "Gipuzkoa", "San Sebastián"],
    "    ↳ Vizcaya / Bizkaia": ["Vizcaya", "Bizkaia", "Bilbao"],
    "📍 La Rioja": ["La Rioja", "Logroño"],
    "📍 Ceuta": ["Ceuta"],
    "📍 Melilla": ["Melilla"],
    "🌐 Todas las CCAA / Ubicaciones": "",
    "📍 Andalucía": "Andalucía",
    "📍 Aragón": "Aragón",
    "📍 Asturias": "Asturias",
    "📍 Illes Balears / Islas Baleares": "Baleares",
    "📍 Canarias": "Canarias",
    "📍 Cantabria": "Cantabria",
    "📍 Castilla-La Mancha": "Castilla-La Mancha",
    "📍 Castilla y León": "Castilla y León",
    "📍 Cataluña / Catalunya": "Cataluña",
    "📍 Comunitat Valenciana": "Valenciana",
    "📍 Extremadura": "Extremadura",
    "📍 Galicia": "Galicia",
    "📍 Madrid (Comunidad de)": "Madrid",
    "📍 Murcia (Región de)": "Murcia",
    "📍 Navarra": "Navarra",
    "📍 País Vasco / Euskadi": "País Vasco",
    "📍 La Rioja": "La Rioja",
    "📍 Ceuta": "Ceuta",
    "📍 Melilla": "Melilla",
}

SECTORES_CPV = {
    "🌐 Todos los sectores CPV": [],
    "Agricultura, alimentación y materias primas (Div. 03-16)": [
        "03",
        "09",
        "14",
        "15",
        "16",
    ],
    "Textil, industria, maquinaria y bienes de consumo (Div. 18-39)": [
        "18",
        "19",
        "22",
        "24",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "37",
        "38",
        "39",
    ],
    "Construcción, agua y energía (Div. 41-48)": [
        "41",
        "42",
        "43",
        "44",
        "45",
        "48",
    ],
    "Servicios generales a empresas y mantenimiento (Div. 50-55)": [
        "50",
        "51",
        "55",
    ],
    "Transporte, correos y telecomunicaciones (Div. 60-65)": [
        "60",
        "63",
        "64",
        "65",
    ],
    "Servicios financieros, inmobiliarios y profesionales (Div. 66-79)": [
        "66",
        "70",
        "71",
        "72",
        "73",
        "75",
        "76",
        "77",
        "79",
    ],
    "Educación, sanidad, medio ambiente y servicios sociales (Div. 80-98)": [
        "80",
        "85",
        "90",
        "92",
        "98",
    ],
    "🌐 Todos los sectores CPV": "",
}

# 4. Interfaz Visual
st.title("🔍 Buscador inteligente de Licitaciones")

if "resultados_acumulados" not in st.session_state:
    st.session_state.resultados_acumulados = []
if "offset_actual" not in st.session_state:
    st.session_state.offset_actual = 0
if "hay_mas_registros" not in st.session_state:
    st.session_state.hay_mas_registros = True


def limpiar_campos():
    st.session_state.consulta_texto = ""
    st.session_state.filtro_fuente = "🌐 Todas las fuentes"
    st.session_state.filtro_ccaa = "🌐 Todas las CCAA / Ubicaciones"
    st.session_state.filtro_lugar_libre = ""
    st.session_state.filtro_cpv_sector = "🌐 Todos los sectores CPV"
    st.session_state.filtro_cpv_codigo = ""
    st.session_state.importe_min = 0.0
    st.session_state.importe_max = 0.0
    st.session_state.resultados_acumulados = []
    st.session_state.offset_actual = 0
    st.session_state.hay_mas_registros = True


consulta_texto = st.text_input(
    "¿Qué tipo de licitación buscas?",
    placeholder="ej. mantenimiento informático, suministro de vehículos, obras...",
    key="consulta_texto",
)

st.markdown("### ⚙️ Filtros avanzados")
col0, col1, col2, col3, col4 = st.columns(5)

with col0:
    try:
        resp_fuentes = supabase.table("licitaciones").select("fuente").execute()
        lista_fuentes_db = sorted(list(set(item["fuente"] for item in resp_fuentes.data if item.get("fuente"))))
    except Exception:
        lista_fuentes_db = []
    opciones_fuente = ["🌐 Todas las fuentes"] + lista_fuentes_db
    filtro_fuente = st.selectbox("📂 Fuente", opciones_fuente, key="filtro_fuente")

with col1:
    importe_min = st.number_input("Importe Mínimo (€)", value=0.0, key="importe_min")
with col2:
    importe_max = st.number_input("Importe Máximo (€)", value=0.0, key="importe_max")
with col3:
    lista_ccaa = list(MAPA_TERRITORIAL.keys())
    filtro_ccaa = st.selectbox("📍 Lugar (CCAA)", lista_ccaa, key="filtro_ccaa")
with col4:
    filtro_lugar_libre = st.text_input("📍 Lugar (Libre)", placeholder="ej. San Sebastián", key="filtro_lugar_libre")

col5, col6, _ = st.columns(3)
with col5:
    lista_sectores = list(SECTORES_CPV.keys())
    filtro_cpv_sector = st.selectbox("📦 Sector CPV", lista_sectores, key="filtro_cpv_sector")
with col6:
    filtro_cpv_codigo = st.text_input("🔢 Código CPV", placeholder="ej. 45210000", key="filtro_cpv_codigo")

col_chk1, col_chk2 = st.columns([2, 2])
with col_chk1:
    usar_filtro_fechas = st.checkbox("📅 Rango fecha publicación", key="usar_filtro_fechas")
with col_chk2:
    usar_filtro_cierre = st.checkbox("⏳ Fecha fin de presentación", key="usar_filtro_cierre")

f_inicio_str, f_fin_str = "", ""
if usar_filtro_fechas:
    st.markdown("##### Rango de fecha de publicación")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_inicio = st.date_input("Desde", value=date(2026, 1, 1))
        f_inicio_str = f_inicio.isoformat()
    with col_f2:
        f_fin = st.date_input("Hasta", value=date(2026, 12, 31))
        f_fin_str = f_fin.isoformat()

fecha_cierre_tope = None
if usar_filtro_cierre:
    st.markdown("##### Fecha fin de presentación de oferta")
    col_c1, _ = st.columns([1, 1])
    with col_c1:
        fecha_cierre_tope = st.date_input("Fecha tope mínima", value=date(2026, 3, 1))

st.write("")

col_btn_buscar, col_btn_novedades, col_btn_limpiar = st.columns([2, 2, 2])

with col_btn_buscar:
    btn_buscar = st.button("🔍 Buscar licitaciones", type="primary", use_container_width=True)

with col_btn_novedades:
    btn_novedades = st.button("✨ Novedades", type="secondary", use_container_width=True)

with col_btn_limpiar:
    st.button("🔄 Limpiar Filtros", on_click=limpiar_campos, type="secondary", use_container_width=True)


def ejecutar_consulta_rpc(offset):
    # Vector de consulta
    if consulta_texto.strip():
        query_con_prefijo = f"query: {consulta_texto.strip()}"
        vector_query = encoder.encode(query_con_prefijo).tolist()
    else:
        vector_query = [0.0]

    # Resolver lugar definitivo
    lugar_final = filtro_lugar_libre.strip()
    if not lugar_final and filtro_ccaa != "🌐 Todas las CCAA / Ubicaciones":
        lugar_final = MAPA_TERRITORIAL[filtro_ccaa]

    # Resolver CPV definitivo
    cpv_final = filtro_cpv_codigo.strip()
    if not cpv_final and filtro_cpv_sector != "🌐 Todos los sectores CPV":
        cpv_final = SECTORES_CPV[filtro_cpv_sector]

    try:
        response = supabase.rpc(
            "buscar_licitaciones_avanzada",
            {
                "query_embedding": vector_query,
                "match_threshold": 0.3 if consulta_texto.strip() else 0.0,
                "p_fuente": filtro_fuente,
                "p_lugar": lugar_final,
                "p_importe_min": float(importe_min),
                "p_importe_max": float(importe_max),
                "p_cpv_sector": "",
                "p_cpv_codigo": cpv_final,
                "p_fecha_ini": f_inicio_str,
                "p_fecha_fin": f_fin_str,
                "p_fecha_cierre_tope": fecha_cierre_tope.isoformat() if fecha_cierre_tope else None,
                "p_limite": 500,
                "p_offset": offset,
            },
        ).execute()
        return response.data or []
    except Exception as e:
        st.error(f"⚠️ Error en la consulta: {e}")
        return []


# 5. Lógica del Botón Novedades
if btn_novedades:
    with st.spinner("Cargando novedades..."):
        data = obtener_novedades_supabase()
        df = pd.DataFrame(data)
        if not df.empty:
            df = df[(df["es_novedad"] == True) | (df["es_actualizada"] == True)]
            if filtro_fuente != "🌐 Todas las fuentes":
                df = df[df["fuente"] == filtro_fuente]
            st.success(f"Se encontraron {len(df)} licitaciones nuevas o actualizadas.")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay novedades.")

# 6. Lógica de Búsqueda Principal con Paginación de 500 en 500
if btn_buscar:
    st.session_state.offset_actual = 0
    st.session_state.resultados_acumulados = []
    st.session_state.hay_mas_registros = True
    
    with st.spinner("Buscando en la base de datos..."):
        nuevos_datos = ejecutar_consulta_rpc(0)
        st.session_state.resultados_acumulados = nuevos_datos
        if len(nuevos_datos) < 500:
            st.session_state.hay_mas_registros = False

# Botón para cargar más registros si la búsqueda está activa o tiene datos acumulados
if st.session_state.resultados_acumulados:
    df = pd.DataFrame(st.session_state.resultados_acumulados)
    if "similarity" in df.columns:
        df["relevancia"] = (df["similarity"] * 100).round(2)
    else:
        df["relevancia"] = 100.0

    st.success(f"Mostrando {len(df)} licitaciones acumuladas.")
    
    tabla_final = []
    for idx, row in enumerate(df.itertuples(), start=1):
        tabla_final.append({
            "#": idx,
            "Relevancia (%)": f"{getattr(row, 'relevancia', 100.0):.2f} %",
            "Título": row.titulo,
            "Órgano": row.organo,
            "Lugar": getattr(row, "lugar_ejecucion", "No especificado"),
            "Cierre": getattr(row, "fecha_fin", "No especificada"),
            "Fecha Pub.": row.fecha,
            "Importe": f"{row.importe:,.2f} €",
            "Enlace": row.enlace,
        })

    df_final = pd.DataFrame(tabla_final)
    st.dataframe(
        df_final,
        column_config={"Enlace": st.column_config.LinkColumn("Enlace oficial", display_text="Ver licitación 🔗")},
        hide_index=True,
        use_container_width=True,
    )

    if st.session_state.hay_mas_registros:
        if st.button("➕ Cargar siguientes 500 resultados"):
            st.session_state.offset_actual += 500
            with st.spinner("Cargando más licitaciones..."):
                siguientes_datos = ejecutar_consulta_rpc(st.session_state.offset_actual)
                if siguientes_datos:
                    st.session_state.resultados_acumulados.extend(siguientes_datos)
                    if len(siguientes_datos) < 500:
                        st.session_state.hay_mas_registros = False
                    st.rerun()
                else:
                    st.session_state.hay_mas_registros = False
                    st.rerun()
    else:
        st.info("Has llegado al final de los resultados disponibles para esta búsqueda.")
