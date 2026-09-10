from datetime import datetime, date
import os
import requests
import lxml.etree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from supabase import Client, create_client

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cac": "urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2",
    "cac-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2",
    "cbc-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2",
}

ESTADOS_CERRADOS = ["EV", "ADJ", "RES", "ANUL", "FOR", "AS", "RE", "CAN"]
TAMANO_LOTE = 10

# ============================================================
# 2. SESIÓN HTTP ROBUSTA
# ============================================================
def crear_sesion_robusta():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ============================================================
# 3. ELIMINAR MADRID DE UNA FUENTE COMBINADA
# ============================================================
def quitar_fuente_madrid(fuente):
    """
    Elimina 'Comunidad de Madrid' de una cadena de fuentes.
    """
    if not fuente:
        return ""
    fuentes = [f.strip() for f in str(fuente).split(",") if f.strip()]
    fuentes_restantes = [
        f for f in fuentes
        if f.casefold() != "comunidad de madrid".casefold()
    ]
    return ", ".join(fuentes_restantes)

# ============================================================
# 4. AUDITORÍA
# ============================================================
def auditar_licitaciones_madrid():
    hoy = date.today()
    sesion = crear_sesion_robusta()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    total_fuente_madrid_eliminada = 0
    total_errores = 0

    print("🔍 Iniciando auditoría por lotes de licitaciones de la Comunidad de Madrid con fecha de cierre 'No especificada'...\n")

    try:
        response = (
            supabase
            .table("licitaciones")
            .select("id, enlace, titulo, fuente, fecha_fin, es_actualizada")
            .ilike("fuente", "%Comunidad de Madrid%")
            .eq("fecha_fin", "No especificada")
            .execute()
        )
        registros_pendientes = response.data or []
    except Exception as e:
        print(f"❌ Error al consultar Supabase: {e}")
        return

    total_pendientes = len(registros_pendientes)
    print(f"📋 Licitaciones de Madrid pendientes de auditar: {total_pendientes}")

    if not registros_pendientes:
        print("\n🎉 No hay licitaciones de la Comunidad de Madrid pendientes de auditar.")
        return

    for inicio in range(0, total_pendientes, TAMANO_LOTE):
        lote = registros_pendientes[inicio:inicio + TAMANO_LOTE]
        numero_lote = (inicio // TAMANO_LOTE) + 1
        print(f"\n📦 --- Procesando Lote {numero_lote} ({len(lote)} registros) ---")

        ids_a_borrar = []

        for reg in lote:
            rec_id = reg.get("id")
            enlace = reg.get("enlace")
            titulo = reg.get("titulo", "Sin título")
            fuente_actual = reg.get("fuente", "")

            if not enlace:
                total_errores += 1
                print(f"   ⚠️ Registro {rec_id} sin enlace. Se omite.")
                continue

            if "ted.europa.eu" in enlace:
                continue

            try:
                resp = sesion.get(enlace, headers=headers, timeout=12)
                if resp.status_code != 200:
                    print(f"   ⚠️ Enlace no disponible (HTTP {resp.status_code})")
                    total_errores += 1
                    continue

                parser = ET.XMLParser(recover=True)
                root = ET.fromstring(resp.content, parser=parser)

                codigo_estado = "PUB"
                estado_el = root.find(".//cbc-place-ext:ContractFolderStatusCode", NS)
                if estado_el is None:
                    estado_el = root.find(".//cbc:ContractFolderStatusCode", NS)
                if estado_el is not None and estado_el.text:
                    codigo_estado = estado_el.text.strip().upper()

                if codigo_estado in ESTADOS_CERRADOS:
                    nueva_fuente = quitar_fuente_madrid(fuente_actual)
                    if not nueva_fuente:
                        ids_a_borrar.append(rec_id)
                        print(f"   🗑️ [A BORRAR - Estado {codigo_estado}]: {titulo[:50]}...")
                        total_eliminadas += 1
                    else:
                        try:
                            supabase.table("licitaciones").update({
                                "fuente": nueva_fuente,
                                "es_actualizada": True
                            }).eq("id", rec_id).execute()
                            print(f"   🔄 [MADRID ELIMINADO DE FUENTE - Estado {codigo_estado}]: {titulo[:50]}...")
                            total_fuente_madrid_eliminada += 1
                            total_actualizadas += 1
                        except Exception as e:
                            print(f"   ❌ Error actualizando fuente del registro {rec_id}: {e}")
                            total_errores += 1
                    continue

                end_date_el = root.find(".//cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod/cbc:EndDate", NS)
                nueva_fecha_fin = "No especificada"
                if end_date_el is not None and end_date_el.text:
                    nueva_fecha_fin = end_date_el.text.strip()[:10]

                if nueva_fecha_fin != "No especificada":
                    try:
                        f_fin_date = datetime.strptime(nueva_fecha_fin, "%Y-%m-%d").date()
                        if f_fin_date < hoy:
                            nueva_fuente = quitar_fuente_madrid(fuente_actual)
                            if not nueva_fuente:
                                ids_a_borrar.append(rec_id)
                                print(f"   🗑️ [A BORRAR - Caducada]: {titulo[:50]}...")
                                total_eliminadas += 1
                            else:
                                try:
                                    supabase.table("licitaciones").update({
                                        "fuente": nueva_fuente,
                                        "es_actualizada": True
                                    }).eq("id", rec_id).execute()
                                    print(f"   🔄 [MADRID ELIMINADO DE FUENTE - Caducada]: {titulo[:50]}...")
                                    total_fuente_madrid_eliminada += 1
                                    total_actualizadas += 1
                                except Exception as e:
                                    print(f"   ❌ Error actualizando fuente del registro {rec_id}: {e}")
                                    total_errores += 1
                            continue
                    except ValueError:
                        pass

                fecha_fin_actual = reg.get("fecha_fin")
                if nueva_fecha_fin != fecha_fin_actual:
                    supabase.table("licitaciones").update({
                        "fecha_fin": nueva_fecha_fin,
                        "es_actualizada": True
                    }).eq("id", rec_id).execute()
                    print(f"   🔄 [ACTUALIZADA Fecha Fin: {nueva_fecha_fin}]: {titulo[:50]}...")
                    total_actualizadas += 1
                else:
                    total_sin_cambios += 1

                time.sleep(0.2)
            except Exception as e:
                print(f"   ⚠️ Error procesando enlace: {e}")
                total_errores += 1
                continue

        if ids_a_borrar:
            try:
                supabase.table("licitaciones").delete().in_("id", ids_a_borrar).execute()
                print(f"   🗑️ -> {len(ids_a_borrar)} licitaciones eliminadas de Supabase en este lote.")
            except Exception as e:
                print(f"   ❌ Error al eliminar lote en Supabase: {e}")
                total_errores += len(ids_a_borrar)

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL DE LA AUDITORÍA DE MADRID:")
    print(f"   - Pendientes auditadas: {total_pendientes}")
    print(f"   - Eliminadas completamente: {total_eliminadas}")
    print(f"   - Madrid eliminada de fuente combinada: {total_fuente_madrid_eliminada}")
    print(f"   - Actualizadas (nueva fecha): {total_actualizadas}")
    print(f"   - Sin cambios: {total_sin_cambios}")
    print(f"   - Errores/enlaces no disponibles: {total_errores}")
    print("\n✅ ¡Auditoría de licitaciones de la Comunidad de Madrid completada!")

if __name__ == "__main__":
    auditar_licitaciones_madrid()
