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

# --- ESTILOS CSS PERSONALIZADOS PARA DISEÑO Y RECUADROS ---
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
        /* Contenedor con borde elegante para agrupar los controles de resultados */
        .results-container {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .alignment-fix {
            display: flex;
            align-items: center;
            height: 100%;
            font-size: 15px;
            color: #31333F;
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

# Inicializar estados de sesión para persistencia de resultados
if "df_resultados" not in st.session_state:
    st.session_state.df_resultados = None
if "mensaje_estado" not in st.session_state:
    st.session_state.mensaje_estado = ""

# 4. Interfaz Visual y Gestión de Estado
st.title("🔍 Buscador inteligente de Licitaciones")


def limpiar_campos():
    st.session_state.consulta_texto = ""
    st.session_state.filtro_fuente = []
    st.session_state.filtro_tipo_contrato = []
    st.session_state.filtro_ccaa = []
    st.session_state.filtro_lugar_libre = ""
    st.session_state.filtro_cpv_sector = []
    st.session_state.filtro_cpv_codigo = ""
    st.session_state.importe_min = 0.0
    st.session_state.importe_max = 0.0
    st.session_state.limite_resultados = 10
    st.session_state.mostrar_todos = True
    st.session_state.df_resultados = None
    st.session_state.mensaje_estado = ""


# Buscador principal
consulta_texto = st.text_input(
    "¿Qué tipo de licitación buscas?",
    placeholder="ej. mantenimiento informático, suministro de vehículos, obras...",
    key="consulta_texto",
)

# Panel de filtros avanzados
st.markdown("### ⚙️ Filtros avanzados")
col0, col_tipo, col1, col2, col3 = st.columns(5)

with col0:
    filtro_fuente = st.multiselect(
        "🌐 Fuente",
        ["Licitaciones Generales PLACSP", "Licitaciones Agregadas PLACSP", "TED", "PSCP Catalunya", "Euskadi", "Comunidad de Madrid", "Contratación Navarra", "Gobierno de La Rioja"],
        default=[],
        key="filtro_fuente",
    )
with col_tipo:
    filtro_tipo_contrato = st.multiselect(
    "📋 Tipo de contrato",
    [
        "Suministros",
        "Servicios",
        "Obras",
        "Gestión de servicios públicos",
        "Concesión de servicios",
        "Concesión de servicios especiales",
        "Concesión de obras",
        "Concesión de obras públicas",
        "Administrativo especial",
        "Privado",
        "Privado de Administración Pública",
        "Patrimonial",
        "Colaboración Público-Privada",
        "Colaboración entre el sector público y el sector privado",
        "Otra legislación sectorial",
        "Contrato de servicios especiales",
        "Otros"
    ],
    default=[], 
    key="filtro_tipo_contrato"
)

with col1:
    importe_min = st.number_input("Importe Mínimo (€)", value=0.0, key="importe_min")
with col2:
    importe_max = st.number_input("Importe Máximo (€)", value=0.0, key="importe_max")
with col3:
    lista_ccaa = list(MAPA_TERRITORIAL.keys())
    filtro_ccaa = st.multiselect(
        "📍 Lugar de ejecución (Desplegable)", 
        [c for c in lista_ccaa if "Todas" not in c], 
        default=[], 
        key="filtro_ccaa"
    )


# Campos de lugar, sector y código CPV en una misma fila
col_lugar, col_sector, col_cpv = st.columns([1, 1, 1])

with col_lugar:
    filtro_lugar_libre = st.text_input(
        "📍 Lugar de ejecución (Libre)",
        placeholder="ej. San Sebastián",
        key="filtro_lugar_libre"
    )

with col_sector:
    lista_sectores = list(SECTORES_CPV.keys())
    filtro_cpv_sector = st.multiselect(
        "📦 Sector CPV",
        [s for s in lista_sectores if "Todos" not in s],
        default=[],
        key="filtro_cpv_sector"
    )

with col_cpv:
    filtro_cpv_codigo = st.text_input(
        "🔢 Código CPV",
        placeholder="ej. 45210000",
        key="filtro_cpv_codigo"
    )

# Fila inferior con fecha fin y rango de publicación alineados en el mismo renglón
# Fila de fechas
col_fecha_fin, col_rango = st.columns([1, 2])

with col_fecha_fin:
    fecha_cierre_tope = st.date_input(
        "⏳ Fecha fin de presentación (Mínima)",
        value=date(2026, 3, 1),
        key="fecha_cierre_tope"
    )

with col_rango:
    # st.markdown("📅 Rango publicación:")

    col_desde, col_hasta = st.columns(2)

    with col_desde:
        f_inicio = st.date_input(
            "📅 Rango publicación (Desde)",
            value=date(2026, 1, 1),
            key="f_inicio"
        )

    with col_hasta:
        f_fin = st.date_input(
            "📅 Rango publicación (Hasta)",
            value=date(2026, 12, 31),
            key="f_fin"
        )
        
# col_f_lbl, col_r_lbl, col_r1, col_r2 = st.columns([1.5, 1.2, 2, 2])
# with col_f_lbl:
#    fecha_cierre_tope = st.date_input(
#        "⏳ Fecha fin de presentación (Mínima)", value=date(2026, 3, 1), key="fecha_cierre_tope"
#    )
#with col_r_lbl:
#    st.markdown('<div class="alignment-fix">📅 Rango publicación:</div>', unsafe_allow_html=True)
#with col_r1:
#    f_inicio = st.date_input("Desde", value=date(2026, 1, 1), key="f_inicio", label_visibility="collapsed")
#with col_r2:
#    f_fin = st.date_input("Hasta", value=date(2026, 12, 31), key="f_fin", label_visibility="collapsed")

# Sección de cantidad, pregunta y barra de resultados metidas dentro de un único recuadro unificado
col_resultados, col_vacio = st.columns([50, 50])

with col_resultados:
    with st.container(border=True):

        st.markdown(
            "<strong>¿Cuántos resultados quieres ver?</strong>",
            unsafe_allow_html=True
        )

        col_res_chk, col_res_texto, col_res_slider = st.columns([2.5, 2, 4])

        with col_res_chk:
            mostrar_todos = st.checkbox(
                "Mostrar TODOS los resultados",
                value=True,
                key="mostrar_todos"
            )

        with col_res_texto:
            st.markdown(
                '<div style="font-size: 15px; padding-top: 8px;">'
                'Seleccionar número de resultados:'
                '</div>',
                unsafe_allow_html=True
            )

        with col_res_slider:
            limite_resultados = st.slider(
                "Resultados",
                min_value=1,
                max_value=500,
                value=10,
                key="limite_resultados",
                label_visibility="collapsed"
            )



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


# Función genérica para aplicar todos los filtros de Pandas en común
def aplicar_filtros_comunes(df):
    if df.empty:
        return df

    # 1. Filtro de fuente flexible para múltiples selecciones
    #if filtro_fuente:
     #   patron_fuentes = "|".join([r"\b" + f + r"\b" for f in filtro_fuente])
     #   df = df[df["fuente"].str.contains(patron_fuentes, case=False, na=False, regex=True)]
    # 1. Filtro de fuente
    # 1. Filtro de fuente - DEBUG
    if filtro_fuente:
        st.write("FILTRO SELECCIONADO:", [repr(f) for f in filtro_fuente])
    
        fuentes_navarra = df[
            df["fuente"]
            .astype(str)
            .str.contains("Navarra", case=False, na=False)
        ]["fuente"].unique()
    
        st.write("FUENTES CON NAVARRA:", [repr(f) for f in fuentes_navarra])
    
        fuentes_seleccionadas = [
            f.strip().casefold()
            for f in filtro_fuente
        ]
    
        df = df[
            df["fuente"]
            .fillna("")
            .astype(str)
            .apply(
                lambda x: any(
                    fuente in x.casefold()
                    for fuente in fuentes_seleccionadas
                )
            )
        ]

    # 2. Filtro de tipo de contrato flexible
    if filtro_tipo_contrato: 
        if "tipo_contrato" in df.columns:
            patron_tipos = "|".join([r"\b" + t + r"\b" for t in filtro_tipo_contrato])
            df = df[df["tipo_contrato"].str.contains(patron_tipos, case=False, na=False, regex=True)]

    # 3. Importes
    if importe_min > 0:
        df = df[df["importe"] >= importe_min]
    if importe_max > 0:
        df = df[df["importe"] <= importe_max]

    # 4. CCAA
    if filtro_ccaa:
        palabras_clave_totales = []
        patrones = []
        for item_ccaa in filtro_ccaa:
            palabras_clave = MAPA_TERRITORIAL.get(item_ccaa, [item_ccaa])
            for p in palabras_clave:
                if p == "Palma":
                    patrones.append(r"(?<![Ll][Aa]\s)\bPalma\b")
                else:
                    patrones.append(r"\b" + p + r"\b")
        patron_regex = "|".join(patrones)
        df = df[df["lugar_ejecucion"].str.contains(patron_regex, case=False, na=False, regex=True)]

    # 5. Lugar libre
    if filtro_lugar_libre.strip():
        df = df[df["lugar_ejecucion"].str.contains(filtro_lugar_libre.strip(), case=False, na=False)]

    # 6. Sector CPV
    if filtro_cpv_sector:
        prefijos_validos = []
        for s in filtro_cpv_sector:
            prefijos_validos.extend(SECTORES_CPV[s])
        prefijos_validos = tuple(prefijos_validos)
        
        def coincide_cpv(cpv_str):
            if not cpv_str or pd.isna(cpv_str) or cpv_str == "No especificado":
                return False
            lista_cpv = [c.strip() for c in str(cpv_str).split(",")]
            return any(c.startswith(prefijos_validos) for c in lista_cpv)
            
        if "cpv" in df.columns:
            df = df[df["cpv"].apply(coincide_cpv)]

    # 7. Código CPV específico
    if filtro_cpv_codigo.strip():
        codigo_busqueda = filtro_cpv_codigo.strip()
        def coincide_codigo_cpv(cpv_str):
            if not cpv_str or pd.isna(cpv_str) or cpv_str == "No especificado":
                return False
            lista_cpv = [c.strip() for c in str(cpv_str).split(",")]
            return any(codigo_busqueda in c for c in lista_cpv)
        if "cpv" in df.columns:
            df = df[df["cpv"].apply(coincide_codigo_cpv)]

    # 8. Fechas de cierre
    def filtrar_fecha_fin(f_str):
        if not f_str:
            return False
        try:
            return date.fromisoformat(f_str[:10]) >= fecha_cierre_tope
        except ValueError:
            return False
    if "fecha_fin" in df.columns:
        df = df[df["fecha_fin"].apply(filtrar_fecha_fin)]

    # 9. Fechas de publicación
    def filtrar_fecha_pub(f_str):
        if not f_str:
            return False
        try:
            return f_inicio <= date.fromisoformat(f_str[:10]) <= f_fin
        except ValueError:
            return False
    if "fecha" in df.columns:
        df = df[df["fecha"].apply(filtrar_fecha_pub)]

    return df


# 5. Lógica del Botón de Novedades
if btn_novedades:
    with st.spinner("Buscando en novedades y actualizaciones..."):
        resultados = []
        consulta_texto_val = consulta_texto if 'consulta_texto' in locals() else ""

        if consulta_texto_val.strip():
            query_con_prefijo = f"query: {consulta_texto_val.strip()}"
            vector_query = encoder.encode(query_con_prefijo).tolist()

            try:
                response = supabase.rpc(
                    "buscar_licitaciones",
                    {
                        "query_embedding": vector_query,
                        "match_threshold": 0.2,
                        "match_count": 999999
                    },
                ).execute()
                resultados = response.data
            except Exception as e:
                st.error(f"⚠️ Error al ejecutar la búsqueda vectorial en novedades: {e}")
        else:
            todos_los_datos = []
            tamano_lote = 1000
            inicio = 0

            while True:
                response = (
                    supabase.table("licitaciones")
                    .select(
                        "titulo, organo, fecha, importe, enlace, lugar_ejecucion,"
                        " fecha_fin, texto_completo, cpv, fuente, tipo_contrato, es_novedad, es_actualizada"
                    )
                    .or_("es_novedad.eq.true,es_actualizada.eq.true")
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

            resultados = todos_los_datos
            for r in resultados:
                r["similarity"] = 1.0

        if not resultados:
            st.warning("No hay nuevas licitaciones ni actualizaciones en este ciclo.")
            st.session_state.df_resultados = None
            st.session_state.mensaje_estado = ""
        else:
            df = pd.DataFrame(resultados)
            
            if "es_novedad" in df.columns and "es_actualizada" in df.columns:
                df = df[(df["es_novedad"] == True) | (df["es_actualizada"] == True)]

            if "similarity" in df.columns:
                df["relevancia"] = (df["similarity"] * 100).round(2)
            else:
                df["relevancia"] = 100.0
                
            df = aplicar_filtros_comunes(df)

            if df.empty:
                st.warning("No hay novedades ni actualizaciones que coincidan con los filtros y la búsqueda indicada.")
                st.session_state.df_resultados = None
                st.session_state.mensaje_estado = ""
            else:
                if not mostrar_todos:
                    total_encontrados = len(df)
                    df = df.head(limite_resultados)
                    mostrados = len(df)

                    if total_encontrados > mostrados:
                        st.session_state.mensaje_estado = f"¡Mostrando las **{mostrados} licitaciones más relevantes** de un total de **{total_encontrados}** encontradas!"
                    else:
                        st.session_state.mensaje_estado = f"¡Se han encontrado y mostrado las {mostrados} licitaciones relevantes!"
                else:
                    mostrados = len(df)
                    st.session_state.mensaje_estado = f"¡Se han encontrado y mostrado las {mostrados} licitaciones relevantes!"

                tabla_final = []
                for idx, row in enumerate(df.itertuples(), start=1):
                    tabla_final.append({
                        "#": idx,
                        "Relevancia (%)": f"{getattr(row, 'relevancia', 100.0):.2f} %",
                        "Título": row.titulo,
                        "Órgano": row.organo,
                        "Tipo Contrato": getattr(row, "tipo_contrato", "No especificado"),
                        "Lugar": getattr(row, "lugar_ejecucion", "No especificado"),
                        "Cierre": getattr(row, "fecha_fin", "No especificada"),
                        "Fecha Pub.": row.fecha,
                        "Importe": f"{row.importe:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €",
                        "Enlace": row.enlace,
                        "Es Novedad": getattr(row, "es_novedad", False),
                        "Es Actualizada": getattr(row, "es_actualizada", False),
                    })

                st.session_state.df_resultados = pd.DataFrame(tabla_final)


# 6. Lógica de Búsqueda Principal vía Supabase RPC
elif btn_buscar:
    with st.spinner("Buscando en Supabase..."):
        resultados = []
        consulta_texto_val = consulta_texto if 'consulta_texto' in locals() else ""

        if consulta_texto_val.strip():
            query_con_prefijo = f"query: {consulta_texto_val.strip()}"
            vector_query = encoder.encode(query_con_prefijo).tolist()
            match_count_deseado = 999999

            try:
                response = supabase.rpc(
                    "buscar_licitaciones",
                    {
                        "query_embedding": vector_query,
                        "match_threshold": 0.2,
                        "match_count": match_count_deseado
                    },
                ).execute()
                resultados = response.data
            except Exception as e:
                st.error(f"⚠️ Error al ejecutar la búsqueda vectorial: {e}")
        else:
            todos_los_datos = []
            tamano_lote = 1000
            inicio = 0

            while True:
                query_sup = (
                    supabase.table("licitaciones")
                    .select(
                        "titulo, organo, fecha, importe, enlace, lugar_ejecucion,"
                        " fecha_fin, texto_completo, cpv, fuente, tipo_contrato, es_novedad, es_actualizada"
                    )
                    .order("fecha", desc=True)
                    .range(inicio, inicio + tamano_lote - 1)
                )
                
                response = query_sup.execute()
                filas = response.data
                
                if not filas:
                    break
                
                todos_los_datos.extend(filas)
                if len(filas) < tamano_lote:
                    break
                inicio += tamano_lote

            resultados = todos_los_datos
            for r in resultados:
                r["similarity"] = 1.0

        if not resultados:
            st.warning("No se encontraron resultados que coincidan con la búsqueda.")
            st.session_state.df_resultados = None
            st.session_state.mensaje_estado = ""
        else:
            df = pd.DataFrame(resultados)
            if "similarity" in df.columns:
                df["relevancia"] = (df["similarity"] * 100).round(2)
            else:
                df["relevancia"] = 100.0

            df = aplicar_filtros_comunes(df)

            if df.empty:
                st.warning("No hay licitaciones que coincidan con los filtros y la búsqueda indicada.")
                st.session_state.df_resultados = None
                st.session_state.mensaje_estado = ""
            else:
                if not mostrar_todos:
                    total_encontrados = len(df)
                    df = df.head(limite_resultados)
                    mostrados = len(df)

                    if total_encontrados > mostrados:
                        st.session_state.mensaje_estado = f"¡Mostrando las **{mostrados} licitaciones más relevantes** de un total de **{total_encontrados}** encontradas!"
                    else:
                        st.session_state.mensaje_estado = f"¡Se han encontrado y mostrado las {mostrados} licitaciones relevantes!"
                else:
                    mostrados = len(df)
                    st.session_state.mensaje_estado = f"¡Se han encontrado y mostrado las {mostrados} licitaciones relevantes!"

                tabla_final = []
                for idx, row in enumerate(df.itertuples(), start=1):
                    tabla_final.append({
                        "#": idx,
                        "Relevancia (%)": f"{getattr(row, 'relevancia', 100.0):.2f} %",
                        "Título": row.titulo,
                        "Órgano": row.organo,
                        "Tipo Contrato": getattr(row, "tipo_contrato", "No especificado"),
                        "Lugar": getattr(row, "lugar_ejecucion", "No especificado"),
                        "Cierre": getattr(row, "fecha_fin", "No especificada"),
                        "Fecha Pub.": row.fecha,
                        "Importe": f"{row.importe:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €",
                        "Enlace": row.enlace,
                        "Es Novedad": getattr(row, "es_novedad", False),
                        "Es Actualizada": getattr(row, "es_actualizada", False),
                    })

                st.session_state.df_resultados = pd.DataFrame(tabla_final)


# --- 7. RENDERIZADO PERSISTENTE DE RESULTADOS ---
if st.session_state.df_resultados is not None and not st.session_state.df_resultados.empty:
    if st.session_state.mensaje_estado:
        st.success(st.session_state.mensaje_estado)

    st.markdown("🟢 *Verde*: Licitaciones Nuevas | 🔵 *Azul*: Licitaciones Actualizadas")

    st.dataframe(
        st.session_state.df_resultados.style.apply(estilizar_filas, axis=1),
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
