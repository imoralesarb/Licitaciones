import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client
from datetime import date

# Configuración de página
st.set_page_config(page_title="Buscador Inteligente de Licitaciones", layout="wide")

# Configuración de Supabase y Modelo (con caché)
@st.cache_resource
def init_connection_and_model():
    SUPABASE_URL = "https://qtubvtxwxnwyxwwyrvzw.supabase.co"
    SUPABASE_KEY = "sb_publishable_Qkq39W0KhkPiHqXVupok7w_xhUws0Lr"
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("Cargando modelo de IA en Streamlit...")
    ai_encoder = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')
    return supabase_client, ai_encoder

supabase, encoder = init_connection_and_model()

st.title("🏛️ Buscador Inteligente de Licitaciones (PLACSP)")
st.markdown("Busca contratos de forma semántica y aplica los filtros que necesites en el mismo panel.")

# Buscador principal
query = st.text_input("¿Qué tipo de contrato o servicio estás buscando?", placeholder="Ej: Obras de saneamiento, suministro de mobiliario...")

# Sección de filtros en la misma vista (organizados en columnas)
st.markdown("### ⚙️ Filtros de búsqueda")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    filtro_lugar = st.text_input("📍 Lugar de ejecución", placeholder="Ej: Pontevedra, Madrid...")

with col_f2:
    filtro_fecha_cierre = st.text_input("⏳ Fecha fin de presentación (texto/parcial)", placeholder="Ej: 2026-09")

with col_f3:
    st.markdown("**📅 Intervalo de fecha de publicación**")
    usar_filtro_fechas = st.checkbox("Activar filtro por fecha de publicación")
    
    # Rango de fechas por defecto (año actual 2026)
    f_inicio = st.date_input("Desde", value=date(2026, 1, 1))
    f_fin = st.date_input("Hasta", value=date(2026, 12, 31))

st.markdown("---")

if st.button("Buscar Licitaciones", type="primary"):
    if not query.strip():
        st.warning("Por favor, introduce un término de búsqueda.")
    else:
        with st.spinner("Buscando las mejores coincidencias con IA..."):
            # Generar embedding de la consulta del usuario
            query_vector = encoder.encode(f"query: {query}").tolist()

            try:
                # Llamada a la función RPC de Supabase
                response = supabase.rpc(
                    "match_licitaciones",
                    {
                        "query_embedding": query_vector,
                        "match_threshold": 0.25,
                        "match_count": 30
                    }
                ).execute()
                
                resultados = response.data

                # --- FILTROS APLICADOS EN PYTHON ---
                resultados_filtrados = []
                for r in resultados:
                    # 1. Filtro de lugar de ejecución
                    if filtro_lugar:
                        lugar_reg = r.get("lugar_ejecucion", "")
                        if not lugar_reg or filtro_lugar.lower() not in lugar_reg.lower():
                            continue
                    
                    # 2. Filtro de fecha de cierre (fecha_fin)
                    if filtro_fecha_cierre:
                        cierre_reg = r.get("fecha_fin", "")
                        if not cierre_reg or filtro_fecha_cierre.lower() not in cierre_reg.lower():
                            continue

                    # 3. Filtro por intervalo de fecha de publicación
                    if usar_filtro_fechas:
                        fecha_pub_str = r.get("fecha", "") # Formato "YYYY-MM-DD"
                        if fecha_pub_str:
                            try:
                                fecha_pub_obj = date.fromisoformat(fecha_pub_str[:10])
                                if not (f_inicio <= fecha_pub_obj <= f_fin):
                                    continue
                            except ValueError:
                                continue
                        else:
                            continue

                    resultados_filtrados.append(r)

                # --- MOSTRAR RESULTADOS ---
                if not resultados_filtrados:
                    st.info("No se han encontrado licitaciones que coincidan con tu búsqueda y los filtros aplicados.")
                else:
                    st.success(f"Se han encontrado {len(resultados_filtrados)} licitaciones:")

                    for item in resultados_filtrados:
                        with st.container():
                            st.subheader(item.get("titulo", "Sin título"))
                            
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("🏢 Órgano", item.get("organo", "Desconocido"))
                            with c2:
                                st.metric("💰 Importe", f"{item.get('importe', 0.0):,.2f} €")
                            with c3:
                                st.metric("📍 Lugar", item.get("lugar_ejecucion", "No especificado"))
                            with c4:
                                st.metric("📅 Cierre", item.get("fecha_fin", "No especificada"))

                            st.text(f"Publicado el: {item.get('fecha', 'Desconocida')}")
                            
                            enlace = item.get("enlace", "")
                            if enlace:
                                st.markdown(f"[🔗 Ver licitación oficial en la PLACSP]({enlace})")
                            
                            st.divider()

            except Exception as e:
                st.error(f"Error al realizar la búsqueda en la base de datos: {e}")
