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
TAMANO_LOTE = 10  # Lotes de 10 en 10 para agilizar y evitar bloqueos

def crear_sesion_robusta():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def auditar_licitaciones_abiertas():
    hoy = date.today()
    sesion = crear_sesion_robusta()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    lote_contador = 1

    print("🔍 Iniciando auditoría por lotes de licitaciones con fecha de cierre 'No especificada'...\n")

    while True:
        try:
            # Traemos un lote pequeño de 10 registros pendientes de auditar
            response = (
                supabase.table("licitaciones")
                .select("*")
                .eq("fecha_fin", "No especificada")
                .limit(TAMANO_LOTE)
                .execute()
            )
            registros = response.data
        except Exception as e:
            print(f"❌ Error al consultar Supabase: {e}")
            break

        # Si ya no quedan registros con fecha no especificada, terminamos el bucle
        if not registros:
            print("\n🎉 ¡Proceso finalizado! No quedan más licitaciones pendientes de auditar en este grupo.")
            break

        print(f"\n📦 --- Procesando Lote {lote_contador} ({len(registros)} registros) ---")
        
        ids_a_borrar = []

        for reg in registros:
            rec_id = reg.get("id")
            enlace = reg.get("enlace")
            titulo = reg.get("titulo", "Sin título")

            if not enlace:
                continue

            # Si es un enlace de TED, lo saltamos o dejamos pasar
            if "ted.europa.eu" in enlace:
                continue

            try:
                resp = sesion.get(enlace, headers=headers, timeout=12)
                if resp.status_code != 200:
                    print(f"   ⚠️ Enlace no disponible (HTTP {resp.status_code})")
                    continue

                parser = ET.XMLParser(recover=True)
                root = ET.fromstring(resp.content, parser=parser)
                
                # 1. Buscar el estado actual del expediente
                codigo_estado = "PUB"
                estado_el = root.find(".//cbc-place-ext:ContractFolderStatusCode", NS) or root.find(".//cbc:ContractFolderStatusCode", NS)
                if estado_el is not None and estado_el.text:
                    codigo_estado = estado_el.text.strip().upper()

                # Si el estado es cerrado/adjudicado/anulado
                if codigo_estado in ESTADOS_CERRADOS:
                    ids_a_borrar.append(rec_id)
                    print(f"   🗑️ [A BORRAR - Estado {codigo_estado}]: {titulo[:50]}...")
                    total_eliminadas += 1
                    continue

                # 2. Buscar si ya se ha fijado una fecha de fin en el XML
                end_date_el = root.find(".//cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod/cbc:EndDate", NS)
                nueva_fecha_fin = "No especificada"
                if end_date_el is not None and end_date_el.text:
                    nueva_fecha_fin = end_date_el.text.strip()[:10]

                # 3. Verificar si ha caducado con la nueva fecha encontrada
                if nueva_fecha_fin != "No especificada":
                    try:
                        f_fin_date = datetime.strptime(nueva_fecha_fin, "%Y-%m-%d").date()
                        if f_fin_date < hoy:
                            ids_a_borrar.append(rec_id)
                            print(f"   🗑️ [A BORRAR - Caducada]: {titulo[:50]}...")
                            total_eliminadas += 1
                            continue
                    except ValueError:
                        pass

                # 4. Comprobar si cambió la fecha fin de 'No especificada' a una fecha activa válida
                if nueva_fecha_fin != reg.get("fecha_fin"):
                    reg["fecha_fin"] = nueva_fecha_fin
                    reg["es_actualizada"] = True
                    
                    # Recalcular embedding por si acaso
                    texto_evaluacion = f"passage: Título: {reg.get('titulo')}. Objeto: {reg.get('texto_completo')}. Órgano: {reg.get('organo')}. Importe: {reg.get('importe')} EUR."
                    reg["embedding"] = encoder.encode(texto_evaluacion).tolist()
                    
                    supabase.table("licitaciones").upsert(reg, on_conflict="enlace").execute()
                    print(f"   🔄 [ACTUALIZADA Fecha Fin: {nueva_fecha_fin}]: {titulo[:50]}...")
                    total_actualizadas += 1
                else:
                    total_sin_cambios += 1

                time.sleep(0.2)

            except Exception as e:
                print(f"   ⚠️ Error procesando enlace: {e}")
                continue

        # Borrar en bloque los registros identificados como cerrados/caducados en este lote
        if ids_a_borrar:
            try:
                supabase.table("licitaciones").delete().in_("id", ids_a_borrar).execute()
                print(f"   🗑️ -> ¡{len(ids_a_borrar)} licitaciones eliminadas de Supabase en este lote!")
            except Exception as e:
                print(f"   ❌ Error al eliminar lote en Supabase: {e}")

        lote_contador += 1
        time.sleep(0.5)  # Pausa breve entre lotes

    print("\n" + "=" * 50)
    print("📊 RESUMEN FINAL DE LA AUDITORÍA:")
    print(f"  - Eliminadas (cerradas/caducadas): {total_eliminadas}")
    print(f"  - Actualizadas (con nueva fecha): {total_actualizadas}")
    print(f"  - Sin cambios: {total_sin_cambios}")
    print("✅ ¡Auditoría de licitaciones completada con éxito y sin bloqueos!")

if __name__ == "__main__":
    auditar_licitaciones_abiertas()
