import os
import feedparser
from datetime import datetime
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# 1. Configuración de credenciales desde las Variables de Entorno de GitHub Actions
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Cargar modelo de IA en CPU (rápido y ligero para GitHub Actions)
encoder = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')

FUENTES_ATOM = [
    {
        "nombre": "Licitaciones Generales",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3.atom"
    },
    {
        "nombre": "Licitaciones Agregadas",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_644/licitacionesAgregadas.atom"
    }
]

def sincronizar():
    print("🔄 Iniciando sincronización de feeds ATOM...")
    
    for fuente in FUENTES_ATOM:
        print(f"📥 Leyendo fuente: {fuente['nombre']}...")
        feed = feedparser.parse(fuente["url"])
        
        for entrada in feed.entries:
            enlace = getattr(entrada, "link", "")
            if not enlace:
                continue
                
            titulo = getattr(entrada, "title", "Sin título")
            organo = getattr(entrada, "author", "Desconocido")
            fecha_pub = getattr(entrada, "published", str(datetime.now().date()))[:10]
            
            # Construir texto completo para el buscador inteligente
            texto_completo = f"Título: {titulo}. Órgano: {organo}."
            
            # Verificar si ya existe en Supabase (para Caso 1 y Caso 2)
            resp = supabase.table("licitaciones").select("id, enlace").eq("enlace", enlace).execute()
            
            if not resp.data:
                # --- CASO 1: NUEVA LICITACIÓN ---
                vector = encoder.encode(f"passage: {texto_completo}").tolist()
                nuevo_registro = {
                    "titulo": titulo,
                    "organo": organo,
                    "fecha": fecha_pub,
                    "enlace": enlace,
                    "texto_completo": texto_completo,
                    "embedding": vector,
                    "importe": 0.0, # Ajustar si extraes el importe del summary del ATOM
                    "lugar_ejecucion": "No especificado"
                }
                supabase.table("licitaciones").insert(nuevo_registro).execute()
                print(f"➕ Nueva añadida: {titulo[:30]}...")
            else:
                # --- CASO 2: ACTUALIZACIÓN DE LICITACIÓN YA EXISTENTE ---
                # Actualizamos los campos por si han cambiado en el feed
                datos_actualizados = {
                    "titulo": titulo,
                    "organo": organo,
                    "texto_completo": texto_completo
                }
                supabase.table("licitaciones").update(datos_actualizados).eq("enlace", enlace).execute()
                print(f"🔄 Actualizada: {titulo[:30]}...")

    print("✅ Sincronización ATOM completada con éxito.")

if __name__ == "__main__":
    sincronizar()
