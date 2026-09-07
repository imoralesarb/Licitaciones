from datetime import datetime, date, timedelta
import os
import time
import requests
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small)...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def procesar_lugar_euskadi(lugar_raw):
    """Añade País Vasco al lugar de ejecución detectado."""
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"
    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "País Vasco"

    lugar_lower = lugar_limpio.lower()
    if "país vasco" not in lugar_lower and "euskadi" not in lugar_lower:
        return f"{lugar_limpio}, País Vasco"
    return lugar_limpio


def normalizar_organo(org):
    """Extrae la raíz del órgano eliminando subcategorías tras guiones o sufijos."""
    if not org:
        return ""
    org_limpio = org.split("-")[0].split("—")[0].strip().lower()
    return org_limpio


# ============================================================
# SICRONIZACIÓN
# ============================================================

def sincronizar_licitaciones_euskadi():
    hoy_date = datetime.now().date()
    ayer_date = hoy_date - timedelta(days=1)
    
    base_url = "https://api.euskadi.eus/administration/events"
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    tipo_licitacion_id = "1"  # Contrataciones administrativas

    fecha_actual = ayer_date
    fecha_fin_rango = hoy_date

    results = []
    print(f"Consultando la API de Euskadi por fechas del {fecha_actual} al {fecha_fin_rango}...")

    while fecha_actual <= fecha_fin_rango:
        year = fecha_actual.year
        month = fecha_actual.month
        day = fecha_actual.day

        url = f"{base_url}/v1.0/events/byType/{tipo_licitacion_id}/byDate/{year}/{month}/{day}"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                events = data.get("events", []) if isinstance(data, dict) else data
                for ev in events:
                    results.append((fecha_actual, ev))
        except Exception as e:
            print(f"Error conectando con la API de Euskadi para la fecha {fecha_actual}: {e}")

        fecha_actual += timedelta(days=1)

    print(f"Descargados {len(results)} registros totales de la API de Euskadi.\n")

    # 1. Resetear flags de novedades anteriores para la fuente Euskadi
    print("Reseteando flags de novedades anteriores...")
    try:
        while True:
            res_antiguos = supabase.table("licitaciones").select("id").eq("fuente", "Euskadi").eq("es_novedad", True).limit(200).execute()
            if not res_antiguos.data:
                break
            ids_antiguos = [item["id"] for item in res_antiguos.data]
            
            for i in range(0, len(ids_antiguos), 50):
                lote_ids = ids_antiguos[i:i+50]
                supabase.table("licitaciones").update({
                    "es_novedad": False,
                    "es_actualizada": False
                }).in_("id", lote_ids).execute()
        print("Flags reseteados con éxito.")
    except Exception as e:
        print(f"Aviso al resetear flags: {e}")

    # 2. Cargar registros existentes en Supabase para validar duplicados y actualizaciones
    try:
        existentes_resp = supabase.table("licitaciones").select("*").eq("fuente", "Euskadi").execute()
        registros_db = {item["enlace"]: item for item in existentes_resp.data if "enlace" in item}
        
        mapa_enlaces = {item["enlace"] for item in existentes_resp.data if "enlace" in item}
        registros_existentes = set()
        for item in existentes_resp.data:
            t = str(item.get("titulo", "")).strip().lower()
            o_base = normalizar_organo(item.get("organo", ""))
            if t or o_base:
                registros_existentes.add((t, o_base))

        print(f"Registros cargados desde Supabase para validación: {len(registros_existentes)}")
    except Exception as e:
        print(f"Error conectando con Supabase para lectura: {e}")
        return

    licitaciones_validas = []
    filtrados_caducados = 0
    filtrados_duplicados = 0
    enlaces_ya_procesados_en_sesion = set()
    claves_sesion = set()

    for fecha_ev, aviso in results:
        enlace = aviso.get("urlEs", "")
        codigo_item = aviso.get("record", "")
        if not enlace:
            enlace = f"https://www.contratacion.euskadi.eus/webkpe00-kpeperfi/es/contenidos/anuncio_contratacion/{codigo_item}/es_doc/index.html"

        titulo_str = str(
            aviso.get("nameEs") or
            aviso.get("nameEu") or
            "Sin título"
        ).strip()

        organo_raw = str(aviso.get("adjudicatorEs", "No especificado")).strip()
        organo_str = organo_raw
        organo_base = normalizar_organo(organo_raw)

        fecha_fin_str = "No especificada"
        deadline_raw = aviso.get("endDate")
        if deadline_raw:
            fecha_fin_str = deadline_raw[:10]
            try:
                cierre_date = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                if cierre_date < hoy_date:
                    filtrados_caducados += 1
                    continue
            except ValueError:
                pass

        fecha_pub = str(aviso.get("startDate", ""))[:10]

        # Extracción de importe y CPV desde el detalle
        importe = 0.0
        cpv = "No especificado"

        if codigo_item:
            try:
                url_detalle = f"https://api.euskadi.eus/procurements/contracting-notices/{codigo_item}"
                resp_detalle = requests.get(url_detalle, headers={"Accept": "application/json"}, timeout=5)
                if resp_detalle.status_code == 200:
                    det_data = resp_detalle.json()
                    importe = float(
                        det_data.get("budgetWithoutVAT") or
                        det_data.get("estimatedValue") or
                        det_data.get("budgetWithVAT") or
                        det_data.get("budget") or 0.0
                    )

                    cpv_raw = det_data.get("CPV") or det_data.get("contractingAuthority", {}).get("CPV", "No especificado")
                    if isinstance(cpv_raw, list):
                        cpv_nombres = [c.get("name", "") for c in cpv_raw if isinstance(c, dict) and c.get("name")]
                        cpv = ", ".join(cpv_nombres) if cpv_nombres else str(cpv_raw)
                    elif isinstance(cpv_raw, dict):
                        cpv = cpv_raw.get("name", str(cpv_raw))
                    else:
                        cpv = str(cpv_raw)
            except Exception as ex:
                print(f"Error consultando detalle para {codigo_item}: {ex}")

        lugar_ejecucion = procesar_lugar_euskadi("País Vasco")

        clave_duplicado = (titulo_str.lower(), organo_base)
        if (enlace in mapa_enlaces or
            enlace in enlaces_ya_procesados_en_sesion or
            clave_duplicado in registros_existentes or
            clave_duplicado in claves_sesion):
            filtrados_duplicados += 1
            continue

        enlaces_ya_procesados_en_sesion.add(enlace)
        claves_sesion.add(clave_duplicado)

        es_nuevo = enlace not in registros_db
        es_actualizado = False

        if not es_nuevo:
            reg_antiguo = registros_db[enlace]
            if (reg_antiguo.get("titulo") != titulo_str or 
                reg_antiguo.get("importe") != importe or 
                reg_antiguo.get("fecha_fin") != fecha_fin_str):
                es_actualizado = True

        texto_completo = f"passage: Título: {titulo_str}. Órgano: {organo_str}. CPV: {cpv}. Lugar: {lugar_ejecucion}. Importe: {importe} EUR."
        embedding = encoder.encode(texto_completo).tolist()

        elemento = {
            "titulo": titulo_str,
            "organo": organo_str,
            "fecha": fecha_pub,
            "importe": importe,
            "enlace": enlace,
            "texto_completo": texto_completo,
            "embedding": embedding,
            "fecha_fin": fecha_fin_str,
            "lugar_ejecucion": lugar_ejecucion,
            "cpv": cpv,
            "es_novedad": es_nuevo,
            "es_actualizada": es_actualizado,
            "fuente": "Euskadi"
        }

        licitaciones_validas.append(elemento)

    # 3. Limpieza automática de caducadas por lotes
    try:
        todos_db = supabase.table("licitaciones").select("id, enlace, fecha_fin").eq("fuente", "Euskadi").execute()
        ids_a_borrar = []
        for item in todos_db.data:
            f_fin = item.get("fecha_fin")
            if f_fin and f_fin != "No especificada":
                try:
                    f_cierre = datetime.strptime(f_fin, "%Y-%m-%d").date()
                    if f_cierre < hoy_date:
                        ids_a_borrar.append(item["id"])
                except ValueError:
                    pass
        
        if ids_a_borrar:
            for i in range(0, len(ids_a_borrar), 50):
                lote_ids = ids_a_borrar[i:i+50]
                supabase.table("licitaciones").delete().in_("id", lote_ids).execute()
            print(f"Eliminadas {len(ids_a_borrar)} licitaciones caducadas de Supabase.")
    except Exception as e:
        print(f"Error en la limpieza de caducadas: {e}")

    print(f"\n--- ESTADÍSTICAS EUSKADI ---")
    print(f"Descartados por fecha caducada: {filtrados_caducados}")
    print(f"Duplicados evitados (con órgano normalizado): {filtrados_duplicados}")
    print(f"Licitaciones válidas listas para insertar: {len(licitaciones_validas)}\n")

    # 4. Inserción optimizada por lotes con reintentos
    if licitaciones_validas:
        print("Subiendo licitaciones de Euskadi a Supabase...")
        tamano_lote = 15
        max_intentos = 3
        subidas_exitosas = 0
        total_a_subir = len(licitaciones_validas)

        for i in range(0, total_a_subir, tamano_lote):
            lote = licitaciones_validas[i:i + tamano_lote]
            num_lote = i // tamano_lote + 1
            exito = False

            for intento in range(1, max_intentos + 1):
                try:
                    supabase.table("licitaciones").upsert(lote, on_conflict="enlace").execute()
                    subidas_exitosas += len(lote)
                    print(f"  -> Lote Euskadi {num_lote} procesado con éxito ({len(lote)} registros).")
                    exito = True
                    break
                except Exception as e:
                    print(f"Intento {intento}/{max_intentos} fallido para lote Euskadi {num_lote}: {e}")
                    if intento < max_intentos:
                        time.sleep(2 * intento)
                    else:
                        print(f"Error definitivo al subir lote Euskadi {num_lote}.")

        print(f"¡Sincronización de Euskadi completada con éxito! Se han subido/actualizado {subidas_exitosas} de {total_a_subir} licitaciones.")
    else:
        print("No hay nuevas licitaciones de Euskadi para insertar.")


if __name__ == "__main__":
    sincronizar_licitaciones_euskadi()
