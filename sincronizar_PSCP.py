# Importación de librerías
from datetime import datetime, date, timedelta
import os
import time
from sodapy import Socrata
import pandas as pd
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client


# Configuración de la Base de Datos

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Carga del modelo de embbedings

print("Cargando modelo de IA (multilingual-e5-small)...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

# Función para añadir Cataluña como lugar de ejecución para facilitar el filtrado
def procesar_lugar(lugar_raw):
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"
    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "Cataluña"
    if "cataluña" not in lugar_limpio.lower() and "catalunya" not in lugar_limpio.lower():
        return f"{lugar_limpio}, Cataluña"
    return lugar_limpio


# Función para sincronizar
def sincronizar_licitaciones_pscp():
    hoy_date = datetime.now().date()
    ayer_date = hoy_date - timedelta(days=1)
    
    # Rango dinámico: desde ayer hasta mañana (dos días)
    fecha_inicio = ayer_date.strftime("%Y-%m-%dT00:00:00")
    fecha_fin = (hoy_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    dataset_id = "ybgg-dgi6"
    client = Socrata("analisi.transparenciacatalunya.cat", None)

    # filtrado por rango de fechas y 'anuncio de licitación'
    query = f"data_publicacio_anunci >= '{fecha_inicio}' AND data_publicacio_anunci < '{fecha_fin}' AND fase_publicacio = 'Anunci de licitació'"

    print("Consultando la API de la PSCP...")
    chunk_size = 1000
    offset = 0
    results = []

    # bucle de paginación
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

    # 1. Marcar todas las anteriores de la bbdd cuya fuente es pscp cataluña como no novedad y no actualizadas
    try:
        supabase.table("licitaciones").update({
            "es_novedad": False,
            "es_actualizada": False
        }).eq("es_novedad", True).eq("fuente", "PSCP Catalunya").execute()
        print("Flags reseteados con éxito.")
    except Exception as e:
        print(f"Error al resetear flags: {e}")
 
    # 2. Cargar registros existentes en Supabase para validar duplicados y actualizaciones
    try:
        existentes_resp = supabase.table("licitaciones").select("*").eq("fuente", "PSCP Catalunya").execute()
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

        if enlace in enlaces_procesados_sesion:
            continue
        enlaces_procesados_sesion.add(enlace)

        texto_completo = f"passage: Título: {titulo_str}. Órgano: {organo_str}. CPV: {cpv}. Lugar: {lugar_ejecucion}. Importe: {importe} EUR."
        
        # Comprobar si ya existe para ver si cambió algo (actualización) o es nuevo
        es_nuevo = enlace not in registros_db
        es_actualizado = False

        if not es_nuevo:
            reg_antiguo = registros_db[enlace]
            if (reg_antiguo.get("titulo") != titulo_str or 
                reg_antiguo.get("importe") != importe or 
                reg_antiguo.get("fecha_fin") != fecha_fin_str):
                es_actualizado = True

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
            "fuente": "PSCP Catalunya"
        }

        licitaciones_validas.append(elemento)

    # 3. Limpieza automática: Borrar de Supabase las que ya estén cerradas o cuya fecha fin ya pasó
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

    # 4. Inserción o actualización en lotes
    if licitaciones_validas:
        print("Subiendo licitaciones a Supabase...")
        tamano_lote = 15
        for i in range(0, len(licitaciones_validas), tamano_lote):
            lote = licitaciones_validas[i:i + tamano_lote]
            try:
                supabase.table("licitaciones").upsert(lote, on_conflict="enlace").execute()
            except Exception as e:
                print(f"Error al subir lote: {e}")
        print("Sincronización completada con éxito.")
    else:
        print("No hay licitaciones para procesar en este rango.")

if __name__ == "__main__":
    sincronizar_licitaciones_pscp()
