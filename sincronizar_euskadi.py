from datetime import datetime, date, timedelta
import os
import time
import re
import requests
from bs4 import BeautifulSoup
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

def limpiar_importe_html(texto):
    """Limpia cadenas de texto de importes forzando estrictamente el formato europeo:
        Punto (.) = separador de miles.
        Coma (,) = separador decimal.
        Ejemplos: '19.880,50' -> 19880.5 | '8.000' -> 8000.0 | '7.232,2' -> 7232.2
    """
    try:
        if not texto:
            return 0.0
        # Si ya es un número directo (int o float), lo devolvemos como float directamente
        if isinstance(texto, (int, float)):
            return float(texto)

        # Eliminar cualquier caracter que no sea dígito, punto, coma o signo menos
        texto_limpio = re.sub(r'[^\d,\.-]', '', str(texto)).strip()
        if not texto_limpio:
            return 0.0

        # Si tiene coma, la coma es el separador decimal europeo y los puntos son miles
        if ',' in texto_limpio:
            texto_limpio = texto_limpio.replace('.', '')  # Quitamos los puntos de miles
            texto_limpio = texto_limpio.replace(',', '.')  # Cambiamos la coma decimal por punto
        elif '.' in texto_limpio:
            # Si solo tiene puntos, comprobamos si actúa como miles (ej: '8.000') o decimal puro (ej: '8.50')
            partes = texto_limpio.split('.')
            if len(partes[-1]) == 3 and len(partes) > 1:
                # Es un punto de miles sin decimales (ej: '8.000')
                texto_limpio = texto_limpio.replace('.', '')
            # Si tiene decimales con punto (ej: '8.50'), se deja como está

        return float(texto_limpio)
    except ValueError:
        return 0.0


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
            res_antiguos = supabase.table("licitaciones").select("id").ilike("fuente", "%Euskadi%").eq("es_novedad", True).limit(200).execute()
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
        print(f"Aviso al resetear flags novedad: {e}")

    try:
        while True:
            res_antiguos = supabase.table("licitaciones").select("id").ilike("fuente", "%Euskadi%").eq("es_actualizada", True).limit(200).execute()
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
        print(f"Aviso al resetear flags actualizada: {e}")

    # 2. Cargar registros existentes en Supabase para validar duplicados y mapear fuentes
    try:
        existentes_resp = supabase.table("licitaciones").select("id, enlace, titulo, organo, fuente").execute()
        
        mapa_enlaces = {}
        registros_existentes = set()
        
        for item in existentes_resp.data:
            enlace_item = item.get("enlace")
            if enlace_item:
                mapa_enlaces[enlace_item] = item

            t = str(item.get("titulo", "")).strip().lower()
            o_base = normalizar_organo(item.get("organo", ""))
            if t or o_base:
                registros_existentes.add((t, o_base))

        print(f"Registros cargados desde Supabase para validación: {len(existentes_resp.data)}")
    except Exception as e:
        print(f"Error conectando con Supabase para lectura: {e}")
        return

    licitaciones_validas = []
    filtrados_caducados = 0
    filtrados_duplicados = 0
    registros_actualizados_count = 0
    enlaces_ya_procesados_en_sesion = set()
    claves_sesion = set()

    for fecha_ev, aviso in results:
        enlace = aviso.get("urlEs") or aviso.get("mainEntityOfPage", "")
        codigo_item = aviso.get("record", "") or aviso.get("id", "")
        if not enlace:
            enlace = f"https://www.contratacion.euskadi.eus/webkpe00-kpeperfi/es/contenidos/anuncio_contratacion/{codigo_item}/es_doc/index.html"

        titulo_str = str(
            aviso.get("object") or
            aviso.get("nameEs") or
            aviso.get("nameEu") or
            "Sin título"
        ).strip()

        organo_raw = str(aviso.get("adjudicatorEs") or aviso.get("socialReason") or "No especificado").strip()
        
        # Extracción y limpieza segura del importe base
        importe_raw = aviso.get("budgetWithoutVAT") or aviso.get("awardAmountWithoutVAT") or 0.0
        importe = limpiar_importe_html(importe_raw)

        tipo_contrato = "No especificado"
        cpv = "No especificado"

        fecha_fin_str = "No especificada"
        deadline_raw = aviso.get("endDate") or aviso.get("contractEndDate")
        if deadline_raw:
            fecha_fin_str = deadline_raw[:10]
            try:
                cierre_date = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                if cierre_date < hoy_date:
                    filtrados_caducados += 1
                    continue
            except ValueError:
                pass

        fecha_pub = str(aviso.get("startDate") or aviso.get("awardDate") or "")[:10]

        # Extracción de importe, CPV y tipo de contrato desde el detalle
        if codigo_item:
            try:
                url_detalle = f"https://api.euskadi.eus/procurements/contracting-notices/{codigo_item}"
                resp_detalle = requests.get(url_detalle, headers={"Accept": "application/json"}, timeout=5)
                if resp_detalle.status_code == 200:
                    det_data = resp_detalle.json()

                    if det_data.get("object"):
                        titulo_str = str(det_data.get("object")).strip()

                    auth_name = det_data.get("contractingAuthority", {}).get("name")
                    org_name = det_data.get("entity", {}).get("org", {}).get("name")
                    if auth_name:
                        organo_raw = auth_name
                    elif org_name:
                        organo_raw = org_name

                    if det_data.get("budgetWithoutVAT") is not None:
                        importe = limpiar_importe_html(det_data.get("budgetWithoutVAT"))

                    # Obtener tipo de contrato
                    ct_obj = det_data.get("contractType")
                    if isinstance(ct_obj, dict):
                        tipo_contrato = ct_obj.get("name", "No especificado")
                    elif isinstance(ct_obj, str):
                        tipo_contrato = ct_obj

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

        organo_str = organo_raw
        organo_base = normalizar_organo(organo_raw)
        lugar_ejecucion = procesar_lugar_euskadi("País Vasco")

        clave_duplicado = (titulo_str.lower(), organo_base)
        if (enlace in enlaces_ya_procesados_en_sesion or
            clave_duplicado in claves_sesion):
            filtrados_duplicados += 1
            continue

        enlaces_ya_procesados_en_sesion.add(enlace)
        claves_sesion.add(clave_duplicado)

        # Comprobar si ya existía en la base de datos (por enlace o por la tupla título/órgano)
        registro_existente = mapa_enlaces.get(enlace)
        if not registro_existente and clave_duplicado in registros_existentes:
            for item_b in mapa_enlaces.values():
                t_b = str(item_b.get("titulo", "")).strip().lower()
                o_b = normalizar_organo(item_b.get("organo", ""))
                if (t_b, o_b) == clave_duplicado:
                    registro_existente = item_b
                    break

        # Regla de fuentes y novedad solicitada
        if registro_existente:
            fuente_actual = str(registro_existente.get("fuente", ""))
            if "euskadi" not in fuente_actual.lower():
                fuente_final = f"{fuente_actual}, euskadi" if fuente_actual else "euskadi"
            else:
                fuente_final = fuente_actual
            
            try:
                supabase.table("licitaciones").update({
                    "fuente": fuente_final
                }).eq("id", registro_existente["id"]).execute()
                registros_actualizados_count += 1
            except Exception as e:
                print(f"Error actualizando fuente para el registro existente {registro_existente.get('id')}: {e}")
            
            continue  
        else:
            fuente_final = "Euskadi"
            es_nuevo = True

        texto_completo = f"passage: Título: {titulo_str}. Órgano: {organo_str}. CPV: {cpv}. Tipo de contrato: {tipo_contrato}. Lugar: {lugar_ejecucion}. Importe: {importe} EUR."
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
            "tipo_contrato": tipo_contrato,
            "es_novedad": es_nuevo,
            "es_actualizada": False,
            "fuente": fuente_final
        }

        licitaciones_validas.append(elemento)

    # 3. Limpieza automática de caducadas por lotes
    try:
        todos_db = supabase.table("licitaciones").select("id, enlace, fecha_fin").ilike("fuente", "%Euskadi%").execute()
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
    print(f"Duplicados evitados / registros existentes actualizados en campo 'fuente': {registros_actualizados_count}")
    print(f"Nuevas licitaciones válidas listas para insertar: {len(licitaciones_validas)}\n")

    # 4. Inserción optimizada por lotes con reintentos para registros nuevos
    if licitaciones_validas:
        print("Subiendo nuevas licitaciones de Euskadi a Supabase...")
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

        print(f"¡Sincronización de Euskadi completada con éxito! Se han subido {subidas_exitosas} de {total_a_subir} licitaciones nuevas.")
    else:
        print("No hay nuevas licitaciones de Euskadi para insertar.")


if __name__ == "__main__":
    sincronizar_licitaciones_euskadi()
