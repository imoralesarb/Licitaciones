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

# --- ESTILOS CSS PERSONALIZADOS PARA LOS BOTONES Y LEYENDAS ---
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


# 2. Cargar modelo de IA en caché (solo codifica el texto de búsqueda)
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")


with st.spinner("Cargando modelo de IA..."):
    encoder = cargar_modelo()


# 3. Descarga auxiliar optimizada solo para el botón de Novedades
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


# MAPA TERRITORIAL COMPLETO DE ESPAÑA
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
}

# MAPA DE SECTORES CPV OFICIALES
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
}

# 4. Interfaz Visual y Gestión de Estado
st.title("🔍 Buscador inteligente de Licitaciones")


def limpiar_campos():
    st.session_state.consulta_texto = ""
    st.session_state.filtro_fuente = "🌐 Todas las fuentes"
    st.session_state.filtro_ccaa = "🌐 Todas las CCAA / Ubicaciones"
    st.session_state.filtro_lugar_libre = ""
    st.session_state.filtro_cpv_sector = "🌐 Todos los sectores CPV"
    st.session_state.filtro_cpv_codigo = ""
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
    key="consulta_texto",
)

# Panel de filtros avanzados
st.markdown("### ⚙️ Filtros avanzados")
col0, col1, col2, col3, col4 = st.columns(5)

with col0:
    # Obtener fuentes únicas de Supabase de manera dinámica para el desplegable (con caché o consulta rápida)
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
    filtro_ccaa = st.selectbox(
        "📍 Lugar de ejecución (Desplegable)", lista_ccaa, key="filtro_ccaa"
    )
with col4:
    filtro_lugar_libre = st.text_input(
        "📍 Lugar de ejecución (Libre)",
        placeholder="ej. San Sebastián",
        key="filtro_lugar_libre",
    )

col5, col6, col7 = st.columns(3)
with col5:
    lista_sectores = list(SECTORES_CPV.keys())
    filtro_cpv_sector = st.selectbox(
        "📦 Sector CPV", lista_sectores, key="filtro_cpv_sector"
    )
with col6:
    filtro_cpv_codigo = st.text_input(
        "🔢 Código CPV", placeholder="ej. 45210000", key="filtro_cpv_codigo"
    )
with col7:
    limite_resultados = st.slider(
        "Resultados", min_value=1, max_value=500, value=10, key="limite_resultados"
    )

col_chk1, col_chk2, col_chk3 = st.columns([1, 2, 2])
with col_chk1:
    mostrar_todos = st.checkbox("Mostrar TODOS los resultados (últimos 1000)", key="mostrar_todos")
with col_chk2:
    usar_filtro_fechas = st.checkbox(
        "📅 Rango fecha publicación en plataforma", key="usar_filtro_fechas"
    )
with col_chk3:
    usar_filtro_cierre = st.checkbox(
        "⏳ Fecha fin de presentación de oferta", key="usar_filtro_cierre"
    )

if usar_filtro_fechas:
    st.markdown("##### Rango de fecha de publicación")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_inicio = st.date_input("Desde", value=date(2026, 1, 1))
    with col_f2:
        f_fin = st.date_input("Hasta", value=date(2026, 12, 31))

if usar_filtro_cierre:
    st.markdown(
        "##### Fecha fin de presentación de oferta"
    )
    col_c1, _ = st.columns([1, 1])
    with col_c1:
        fecha_cierre_tope = st.date_input(
            "Fecha tope mínima de fin de presentación (Muestra las que acaban en este día o después)", value=date(2026, 3, 1)
        )

st.write("")

# --- BOTONES DE ACCIÓN PRINCIPAL ---
col_btn_buscar, col_btn_novedades, col_btn_limpiar = st.columns([2, 2, 2])

with col_btn_buscar:
    btn_buscar = st.button(
        "🔍 Buscar licitaciones", type="primary", use_container_width=True
    )

with col_btn_novedades:
    btn_novedades = st.button(
        "✨ Novedades",
        type="secondary",
        use_container_width=True,
    )

with col_btn_limpiar:
    btn_limpiar = st.button(
        "🔄 Limpiar Filtros",
        on_click=limpiar_campos,
        type="secondary",
        use_container_width=True,
    )


# Función auxiliar para pintar las filas de la tabla de Streamlit
def estilizar_filas(row):
    if row.get("Es Novedad", False):
        return [
            "background-color: #d4edda; color: #155724; font-weight: bold;"
        ] * len(row)
    elif row.get("Es Actualizada", False):
        return [
            "background-color: #cce5ff; color: #004085; font-weight: bold;"
        ] * len(row)
    return [""] * len(row)


# 5. Lógica del Botón de Novedades
if btn_novedades:
    with st.spinner("Cargando novedades y actualizaciones..."):
        data = obtener_novedades_supabase()
        if not data:
            st.info("No hay datos en Supabase.")
        else:
            df = pd.DataFrame(data)
            df = df[(df["es_novedad"] == True) | (df["es_actualizada"] == True)]

            # Filtrar por fuente seleccionada si aplica
            if filtro_fuente != "🌐 Todas las fuentes" and "fuente" in df.columns:
                df = df[df["fuente"] == filtro_fuente]

            if df.empty:
                st.info("No hay nuevas licitaciones ni actualizaciones en este ciclo con los filtros seleccionados.")
            else:
                st.success(
                    f"¡Se han encontrado {len(df)} licitaciones nuevas o actualizadas!"
                )
                st.markdown(
                    "🟢 *Verde*: Licitaciones Nuevas | 🔵 *Azul*: Licitaciones"
                    " Actualizadas"
                )

                tabla_final = []
                for idx, row in enumerate(df.itertuples(), start=1):
                    tabla_final.append({
                        "#": idx,
                        "Título": row.titulo,
                        "Órgano": row.organo,
                        "Lugar": getattr(row, "lugar_ejecucion", "No especificado"),
                        "Cierre": getattr(row, "fecha_fin", "No especificada"),
                        "Fecha Pub.": row.fecha,
                        "Importe": f"{row.importe:,.2f} €",
                        "Enlace": row.enlace,
                        "Es Novedad": getattr(row, "es_novedad", False),
                        "Es Actualizada": getattr(row, "es_actualizada", False),
                    })

                df_final = pd.DataFrame(tabla_final)

                st.dataframe(
                    df_final.style.apply(estilizar_filas, axis=1),
                    column_config={
                        "Enlace": st.column_config.LinkColumn(
                            "Enlace oficial", display_text="Ver licitación 🔗"
                        ),
                        "Es Novedad": None,
                        "Es Actualizada": None,
                    },
                    hide_index=True,
                    use_container_width=True,
                )

# 6. Lógica de Búsqueda Principal vía Supabase RPC (Optimizado sin bloquear CPU)
elif btn_buscar:
    with st.spinner("Buscando en Supabase..."):
        resultados = []

        # Comprobamos si hay texto de búsqueda
        if consulta_texto.strip():
            query_con_prefijo = f"query: {consulta_texto.strip()}"
            vector_query = encoder.encode(query_con_prefijo).tolist()
        else:
            # Si no hay texto, usamos un vector neutral (ceros) para que la función RPC 
            # devuelva registros sin depender de una similitud semántica de texto.
            # (Asegúrate de que el tamaño coincida con tu modelo, ej: 384 para multilingual-e5-small)
            vector_query = [0.0] * 384 

        try:
            response = supabase.rpc(
                "buscar_licitaciones",
                {
                    "query_embedding": vector_query,
                    # Si no hay texto, bajamos el umbral a 0 para no descartar nada por similitud
                    "match_threshold": 0.0 if not consulta_texto.strip() else 0.3,
                    "match_count": 999999 if mostrar_todos else 1000,
                },
            ).execute()
            resultados = response.data
        except Exception as e:
            st.error(f"⚠️ Error al ejecutar la búsqueda: {e}")

        if not resultados:
            st.warning("No se encontraron resultados con los filtros indicados.")
        else:
            df = pd.DataFrame(resultados)
            if "similarity" in df.columns and consulta_texto.strip():
                df["relevancia"] = (df["similarity"] * 100).round(2)
            else:
                df["relevancia"] = 100.0

            # --- A partir de aquí siguen tus filtros de Pandas habituales ---
            if not df.empty and filtro_fuente != "🌐 Todas las fuentes" and "fuente" in df.columns:
                df = df[df["fuente"] == filtro_fuente]

            if not df.empty and importe_min > 0:
                df = df[df["importe"] >= importe_min]
            if not df.empty and importe_max > 0:
                df = df[df["importe"] <= importe_max]

            # (Resto de filtros de CCAA, CPV y Fechas que ya tienes...)

            if not df.empty and filtro_ccaa != "🌐 Todas las CCAA / Ubicaciones":
                palabras_clave = MAPA_TERRITORIAL.get(filtro_ccaa, [filtro_ccaa])
                if "Baleares" in filtro_ccaa or "Balears" in filtro_ccaa:
                    patrones = []
                    for p in palabras_clave:
                        if p == "Palma":
                            patrones.append(r"(?<![Ll][Aa]\s)\bPalma\b")
                        else:
                            patrones.append(r"\b" + p + r"\b")
                    patron_regex = "|".join(patrones)
                else:
                    patron_regex = "|".join([r"\b" + p + r"\b" for p in palabras_clave])
                df = df[
                    df["lugar_ejecucion"].str.contains(
                        patron_regex, case=False, na=False, regex=True
                    )
                ]

            if not df.empty and filtro_lugar_libre.strip():
                df = df[
                    df["lugar_ejecucion"].str.contains(
                        filtro_lugar_libre.strip(), case=False, na=False
                    )
                ]

            if not df.empty and filtro_cpv_sector != "🌐 Todos los sectores CPV":
                prefijos_validos = tuple(SECTORES_CPV[filtro_cpv_sector])

                def coincide_cpv(cpv_str):
                    if not cpv_str or pd.isna(cpv_str) or cpv_str == "No especificado":
                        return False
                    lista_cpv = [c.strip() for c in str(cpv_str).split(",")]
                    return any(c.startswith(prefijos_validos) for c in lista_cpv)

                if "cpv" in df.columns:
                    df = df[df["cpv"].apply(coincide_cpv)]

            if not df.empty and filtro_cpv_codigo.strip():
                codigo_busqueda = filtro_cpv_codigo.strip()

                def coincide_codigo_cpv(cpv_str):
                    if not cpv_str or pd.isna(cpv_str) or cpv_str == "No especificado":
                        return False
                    lista_cpv = [c.strip() for c in str(cpv_str).split(",")]
                    return any(codigo_busqueda in c for c in lista_cpv)

                if "cpv" in df.columns:
                    df = df[df["cpv"].apply(coincide_codigo_cpv)]

            if not df.empty and usar_filtro_cierre:

                def filtrar_fecha_fin(f_str):
                    if not f_str:
                        return False
                    try:
                        return date.fromisoformat(f_str[:10]) >= fecha_cierre_tope
                    except ValueError:
                        return False

                if "fecha_fin" in df.columns:
                    df = df[df["fecha_fin"].apply(filtrar_fecha_fin)]

            if not df.empty and usar_filtro_fechas:

                def filtrar_fecha_pub(f_str):
                    if not f_str:
                        return False
                    try:
                        return f_inicio <= date.fromisoformat(f_str[:10]) <= f_fin
                    except ValueError:
                        return False

                if "fecha" in df.columns:
                    df = df[df["fecha"].apply(filtrar_fecha_pub)]

            if not df.empty and not mostrar_todos:
                df = df.head(limite_resultados)

            if df.empty:
                st.warning("No se encontraron resultados con los filtros indicados.")
            else:
                st.success(f"¡Se han encontrado {len(df)} licitaciones relevantes!")
                st.markdown(
                    "🟢 *Verde*: Licitaciones Nuevas | 🔵 *Azul*: Licitaciones"
                    " Actualizadas"
                )

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
                        "Es Novedad": getattr(row, "es_novedad", False),
                        "Es Actualizada": getattr(row, "es_actualizada", False),
                    })

                df_final = pd.DataFrame(tabla_final)

                st.dataframe(
                    df_final.style.apply(estilizar_filas, axis=1),
                    column_config={
                        "Enlace": st.column_config.LinkColumn(
                            "Enlace oficial", display_text="Ver licitación 🔗"
                        ),
                        "Es Novedad": None,
                        "Es Actualizada": None,
                    },
                    hide_index=True,
                    use_container_width=True,
                )
