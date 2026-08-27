import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client

# Configuración de página
st.set_page_config(page_title="Buscador Inteligente de Licitaciones", layout="wide")

# Configuración de Supabase y Modelo (usando caché para optimizar rendimiento)
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
st.markdown("Busca licitaciones de forma semántica y filtra por ubicación o plazos.")

# Barra de búsqueda principal
query = st.text_input("¿Qué tipo de contrato o servicio estás buscando?", placeholder="Ej: Obras de saneamiento, suministro de mobiliario...")

# Barra lateral para filtros avanzados
st.sidebar.header("Filtros Avanzados")
filtro_lugar = st.sidebar.text_input("Filtrar por Lugar (ej: Pontevedra, Madrid...)")
filtro_importe_min = st.sidebar.number_input("Importe mínimo (€)", min_value=0.0, value=0.0, step=1000.0)

if st.button("Buscar Licitaciones", type="primary"):
    if not query.strip():
        st.warning("Por favor, introduce un término de búsqueda.")
    else:
        with st.spinner("Buscando las mejores coincidencias con IA..."):
            # Generar embedding de la consulta del usuario (prefijado para query en multilingual-e5)
            query_vector = encoder.encode(f"query: {query}").tolist()

            # Llamada a la función RPC de Supabase para búsqueda vectorial (ajusta el nombre de tu función si difiere)
            try:
                response = supabase.rpc(
                    "match_licitaciones",
                    {
                        "query_embedding": query_vector,
                        "match_threshold": 0.3,
                        "match_count": 15
                    }
                ).execute()
                
                resultados = response.data

                # Aplicar filtros adicionales en Python si el usuario los indicó
                if filtro_lugar:
                    resultados = [r for r in resultados if r.get("lugar_ejecucion") and filtro_lugar.lower() in r["lugar_ejecucion"].lower()]
                
                if filtro_importe_min > 0:
                    resultados = [r for r in resultados if r.get("importe", 0) >= filtro_importe_min]

                if not resultados:
                    st.info("No se han encontrado licitaciones que coincidan con tu búsqueda y los filtros aplicados.")
                else:
                    st.success(f"Se han encontrado {len(resultados)} licitaciones:")

                    for item in resultados:
                        with st.container():
                            st.subheader(item.get("titulo", "Sin título"))
                            
                            # Organizar métricas o detalles visuales
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🏢 Órgano", item.get("organo", "Desconocido"))
                            with col2:
                                st.metric("💰 Importe", f"{item.get('importe', 0.0):,.2f} €")
                            with col3:
                                st.metric("📍 Lugar", item.get("lugar_ejecucion", "No especificado"))

                            # Información adicional (Fechas y enlace)
                            st.markdown(f"📅 **Fecha límite presentación:** `{item.get('fecha_fin', 'No especificada')}`")
                            
                            enlace = item.get("enlace", "")
                            if enlace:
                                st.markdown(f"[🔗 Ver licitación oficial en la PLACSP]({enlace})")
                            
                            st.divider()

            except Exception as e:
                st.error(f"Error al realizar la búsqueda en la base de datos: {e}")
