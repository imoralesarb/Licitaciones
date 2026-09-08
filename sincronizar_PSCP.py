# -*- coding: utf-8 -*-
from datetime import datetime, date, timedelta
import os
import time
from sodapy import Socrata
import pandas as pd
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


def traducir_tipo_contrato(tipus_cat):
    """Traduce exhaustivamente todos los tipos de contrato de la PSCP al castellano para la BBDD."""
    if not tipus_cat:
        return "No especificado"
    
    limpio = str(tipus_cat).strip().lower()
    
    mapping = {
        "serveis": "Servicios",
        "subministraments": "Suministros",
        "obres": "Obras",
        "no especificado": "No especificado",
        "administratiu especial": "Administrativo especial",
        "concessió de serveis": "Concesión de servicios",
        "altra legislació sectorial": "Otra legislación sectorial",
        "contracte de serveis especials (annex iv)": "Contrato de servicios especiales",
        "privat d'administració pública": "Privado de Administración Pública",
        "concessió d'obres": "Concesión de obras",
        "concessió de serveis especials (annex iv)": "Concesión de servicios especiales",
        "col·laboració públic-privat": "Colaboración Público-Privada"
    }
    
    return mapping.get(limpio, str(tipus_cat).capitalize())


def procesar_lugar(lugar_raw):
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"
    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "Cataluña"
    if "cataluña" not in lugar_limpio.lower() and "catalunya" not in lugar_limpio.lower():
        return f"{lugar_limpio}, Cataluña"
    return lugar_limpio


def sincronizar_licitaciones_pscp():
    hoy_date = datetime.now().date()
    ayer_date = hoy_date - timedelta(days=1)
    
    fecha_inicio = ayer_date.strftime("%Y-%m-%dT00:00:00")
    fecha_fin = (hoy_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    dataset_id = "ybgg-dgi6"
    client = Socrata("analisi.transparenciacatalunya.cat", None)

    query = f"data_publicacio_anunci >= '{fecha_inicio}' AND data_publicacio_anunci < '{fecha_fin}' AND fase_publicacio = 'Anunci de licitació'"

    print("Consultando la API de la PSCP...")
    chunk_size = 1000
    offset = 0
    results = []

    while True:
        try:
            chunk = client.get(
                dataset_id,
                where=query,
                order="data_publicacio_anunci DESC",
                limit=chunk_size,
                offset=offset
            )
            if not chunk:
                break
            results.extend(chunk)
            if len(chunk) < chunk_size:
                break
            offset += chunk_size
        except Exception as e:
            print(f"Error conectando con la API de PSCP: {e}")
            break

    print(f"Total registros obtenidos de la API PSCP: {len(results)}")

    # 1. Resetear flags de novedades anteriores
    print("Reseteando flags de novedades anteriores...")
    try:
        while True:
            res_antiguos = supabase.table("licitaciones").select("id").eq("fuente", "PSCP Catalunya").eq("es_novedad", True).limit(200).execute()
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

    # 2. Cargar todos los registros existentes en Supabase para validar duplicados y actualizar fuentes globales
    try:
        existentes_resp = supabase.table("licitaciones").select("enlace, titulo, organo, fuente, tipo_contrato, fecha_fin, importe").execute()
        registros_db = {item["enlace"]: item for item in existentes_resp.data if "enlace" in item}
    except Exception as e:
        print(f"Error conectando con Supabase para lectura: {e}")
        return

    licitaciones_validas = []
    filtrados_caducados = 0
    enlaces_procesados_sesion = set()

    for aviso in results:
        enlace_raw = aviso.get("enllac_publicacio")
        enlace = enlace_raw.get('url') if isinstance(enlace_raw, dict) and 'url' in enlace_raw else str(enlace_raw)

        if not enlace:
            continue

        titulo_str = str(aviso.get("denominacio", "Sin título")).strip()
        organo_str = str(aviso.get("nom_organ", "No especificado")).strip()

        fecha_fin_str = "No especificada"
        fecha_cierre_raw = aviso.get("termini_presentacio_ofertes")
        if fecha_cierre_raw:
            fecha_fin_str = fecha_cierre_raw[:10]
            try:
                cierre_date = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                if cierre_date < hoy_date:
                    filtrados_caducados += 1
                    continue
            except ValueError:
                pass

        fecha_pub = aviso.get("data_publicacio_anunci", "")[:10]
        importe_val = aviso.get("pressupost_licitacio_sense_iva_expedient") or aviso.get("pressupost_licitacio_sense_1") or 0.0
        try:
            importe = float(importe_val)
        except (ValueError, TypeError):
            importe = 0.0

        cpv_raw = aviso.get("codi_cpv", "No especificado")
        cpv = ", ".join([c.strip() for c in str(cpv_raw).split("||") if c.strip()]) if cpv_raw and cpv_raw != "No especificado" else "No especificado"
        
        lugar_bruto = aviso.get("lloc_execucio", "No especificado")
        lugar_ejecucion = procesar_lugar(lugar_bruto)

        # Extracción y traducción del tipo de contrato
        tipus_cat = aviso.get("tipus_contracte", "No especificado")
        tipo_contrato = traducir_tipo_contrato(tipus_cat)

        if enlace in enlaces_procesados_sesion:
            continue
        enlaces_procesados_sesion.add(enlace)

        texto_completo = f"passage: Título: {titulo_str}. Órgano: {organo_str}. Tipo de contrato: {tipo_contrato}. CPV: {cpv}. Lugar: {lugar_ejecucion}. Importe: {importe} EUR."
        
        if enlace in registros_db:
            # El registro ya existe en la BBDD: actualizamos fuente y tipo si procede, sin duplicar
            reg_antiguo = registros_db[enlace]
            fuente_actual = reg_antiguo.get("fuente", "")
            tipo_actual = reg_antiguo.get("tipo_contrato", "")
            
            actualizar_datos = {}
            
            # Añadir fuente al final si no la tiene
            if "PSCP Catalunya" not in fuente_actual:
                nueva_fuente = f"{fuente_actual}, PSCP Catalunya" if fuente_actual else "PSCP Catalunya"
                actualizar_datos["fuente"] = nueva_fuente

            # Añadir tipo de contrato si está vacío o no especificado
            if (not tipo_actual or tipo_actual == "No especificado") and tipo_contrato != "No especificado":
                actualizar_datos["tipo_contrato"] = tipo_contrato

            # Detectar si hay cambios para marcar como actualizado
            es_actualizado = (
                reg_antiguo.get("titulo") != titulo_str or 
                reg_antiguo.get("importe") != importe or 
                reg_antiguo.get("fecha_fin") != fecha_fin_str
            )
            if es_actualizado:
                actualizar_datos["es_actualizada"] = True

            if actualizar_datos:
                try:
                    supabase.table("licitaciones").update(actualizar_datos).eq("enlace", enlace).execute()
                except Exception as e:
                    print(f"Error actualizando registro existente {enlace}: {e}")
            continue

        # Registro nuevo
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
            "es_novedad": True,
            "es_actualizada": False,
            "fuente": "PSCP Catalunya"
        }

        licitaciones_validas.append(elemento)

    # 3. Limpieza automática de caducadas por lotes
    try:
        todos_db = supabase.table("licitaciones").select("id, enlace, fecha_fin").eq("fuente", "PSCP Catalunya").execute()
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

    # 4. Inserción optimizada con lotes pequeños y reintentos
    if licitaciones_validas:
        total_a_subir = len(licitaciones_validas)
        print(f"Subiendo un total de {total_a_subir} licitaciones a Supabase...")
        
        tamano_lote = 5
        subidas_exitosas = 0
        max_intentos = 3
        
        for i in range(0, total_a_subir, tamano_lote):
            lote = licitaciones_validas[i:i + tamano_lote]
            num_lote = i // tamano_lote + 1
            exito = False
            
            for intento in range(1, max_intentos + 1):
                try:
                    supabase.table("licitaciones").upsert(lote, on_conflict="enlace").execute()
                    subidas_exitosas += len(lote)
                    print(f"Progreso: {subidas_exitosas}/{total_a_subir} licitaciones procesadas...")
                    exito = True
                    break
                except Exception as e:
                    print(f"⚠️ Intento {intento}/{max_intentos} fallido para lote PSCP {num_lote}: {e}")
                    if intento < max_intentos:
                        time.sleep(2 * intento)
                    else:
                        print(f"❌ Error definitivo al subir lote PSCP {num_lote}.")
                
        print(f"Sincronización completada con éxito. Se han subido/actualizado {subidas_exitosas} de {total_a_subir} licitaciones.")
    else:
        print("No hay licitaciones nuevas para procesar en este rango.")

if __name__ == "__main__":
    sincronizar_licitaciones_pscp()
