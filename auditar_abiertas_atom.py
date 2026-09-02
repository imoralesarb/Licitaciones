from datetime import datetime, date
import os
import requests
import lxml.etree as ET
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client
from urllib3.util.retry import Retry
import time

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small) para auditoría...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cac": "urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2",
    "cac-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2",
    "cbc-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2",
}

ESTADOS_CERRADOS = ["EV", "ADJ", "RES", "ANUL", "FOR", "AS", "RE", "CAN"]

def crear_sesion_robusta():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def auditar_licitaciones_abiertas():
    hoy = date.today()
    
    print("🔍 Buscando en Supabase licitaciones con fecha de cierre 'No especificada'...")
    try:
        response = supabase.table("licitaciones").select("*").eq("fecha_fin", "No especificada").execute()
        registros = response.data
    except Exception as e:
        print(f"❌ Error al consultar Supabase: {e}")
        return

    if not registros:
        print("ℹ️ No hay licitaciones con fecha no especificada para auditar.")
        return

    print(f"📋 Se van a auditar {len(registros)} registros...\n")
    sesion = crear_sesion_robusta()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    eliminadas = 0
    actualizadas = 0
    sin_cambios = 0

    for idx, reg in enumerate(registros):
        enlace = reg.get("enlace")
        if not enlace:
            continue

        # Si es un enlace de TED, la API de TED o su estructura difiere, nos enfocamos principalmente en PLACSP
        # (O puedes adaptarlo si el enlace es de TED). Aquí cubrimos la estructura estándar XML/Atom de PLACSP.
        if "ted.europa.eu" in enlace:
            # Para TED las dejamos pasar o implementas comprobación específica si lo deseas
            continue

        try:
            resp = sesion.get(enlace, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"⚠️ [{idx+1}/{len(registros)}] Enlace no disponible (HTTP {resp.status_code}): {enlace}")
                continue

            parser = ET.XMLParser(recover=True)
            root = ET.fromstring(resp.content, parser=parser)
            
            # Buscar el estado actual del expediente
            codigo_estado = "PUB"
            estado_el = root.find(".//cbc-place-ext:ContractFolderStatusCode", NS) or root.find(".//cbc:ContractFolderStatusCode", NS)
            if estado_el is not None and estado_el.text:
                codigo_estado = estado_el.text.strip().upper()

            # Si el estado es cerrado/adjudicado/anulado, lo borramos
            if codigo_estado in ESTADOS_CERRADOS:
                supabase.table("licitaciones").delete().eq("enlace", enlace).execute()
                print(f"🗑️ [ELIMINADA - Estado {codigo_estado}]: {reg.get('titulo')}")
                eliminadas += 1
                time.sleep(0.5)
                continue

            # Buscar si ya se ha fijado una fecha de fin
            end_date_el = root.find(".//cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod/cbc:EndDate", NS)
            nueva_fecha_fin = "No especificada"
            if end_date_el is not None and end_date_el.text:
                nueva_fecha_fin = end_date_el.text.strip()[:10]

            # Verificar si ha caducado con la nueva fecha
            if nueva_fecha_fin != "No especificada":
                try:
                    f_fin_date = datetime.strptime(nueva_fecha_fin, "%Y-%m-%d").date()
                    if f_fin_date < hoy:
                        supabase.table("licitaciones").delete().eq("enlace", enlace).execute()
                        print(f"🗑️ [ELIMINADA - Caducada hoy]: {reg.get('titulo')}")
                        eliminadas += 1
                        time.sleep(0.5)
                        continue
                except ValueError:
                    pass

            # Comprobar si cambió algún otro campo relevante (ej: título, importe, etc.) por si se actualizó
            # Extraemos los datos actuales de la web para comparar
            titulo_el = root.find("atom:title", NS) or root.find(".//cbc:Title", NS)
            nuevo_titulo = titulo_el.text.strip() if titulo_el is not None and titulo_el.text else reg.get("titulo")

            # Si la fecha fin cambió de 'No especificada' a una fecha válida
            if nueva_fecha_fin != reg.get("fecha_fin"):
                # Actualizamos el registro
                reg["fecha_fin"] = nueva_fecha_fin
                reg["es_actualizada"] = True
                
                # Reconstruir texto completo y embedding por si acaso
                texto_evaluacion = f"passage: Título: {reg.get('titulo')}. Objeto: {reg.get('texto_completo')}. Órgano: {reg.get('organo')}. Importe: {reg.get('importe')} EUR."
                reg["embedding"] = encoder.encode(texto_evaluacion).tolist()
                
                supabase.table("licitaciones").upsert(reg, on_conflict="enlace").execute()
                print(f"🔄 [ACTUALIZADA Fecha Fin a {nueva_fecha_fin}]: {reg.get('titulo')}")
                actualizadas += 1
            else:
                sin_cambios += 1

            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Error procesando enlace {enlace}: {e}")
            continue

    print(f"\n📊 Resumen Auditoría:")
    print(f"  - Eliminadas (cerradas/caducadas): {eliminadas}")
    print(f"  - Actualizadas (con nueva fecha/datos): {actualizadas}")
    print(f"  - Sin cambios: {sin_cambios}")
    print("✅ ¡Auditoría de licitaciones abiertas completada con éxito!")

if __name__ == "__main__":
    auditar_licitaciones_abiertas()
