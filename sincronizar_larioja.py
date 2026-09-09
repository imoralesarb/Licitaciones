# -*- coding: utf-8 -*-
from datetime import datetime, date, timedelta
import os
import time
import requests
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


def procesar_lugar_rioja(lugar_raw):
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"
    if lugar_limpio == "ES230" or not lugar_limpio or lugar_limpio == "No especificado":
        return "La Rioja"
    if "rioja" not in lugar_limpio.lower():
        return f"{lugar_limpio}, La Rioja"
    return lugar_limpio


def sincronizar_licitaciones_rioja():
    url_json = "https://ias1.larioja.org/opendata/download?r=Y2Q9MTc5fGNmPTA0"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01"
    }

    print("Descargando el archivo JSON de contratación del Gobierno de La Rioja...")
    try:
        response = requests.get(url_json, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"Error HTTP {response.status_code} al descargar los datos.")
            return
        data_json = response.json()
        results = data_json.get("data", [])
    except Exception as e:
        print(f"Error conectando con el portal de La Rioja: {e}")
        return

    print(f"Total registros obtenidos de La Rioja: {len(results)}")
    
    hoy_date = datetime.now().date()
    limite_fecha = hoy_date - timedelta(days=2)
    print(f"Filtrando licitaciones publicadas desde: {limite_fecha} hasta {hoy_date}")

    # 1. Resetear flags de novedades anteriores
    print("Reseteando flags de novedades anteriores...")
    try:
        while True:
            res_antiguos = supabase.table("licitaciones").select("id").eq("fuente", "Gobierno de La Rioja").eq("es_novedad", True).limit(200).execute()
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

    try:
        while True:
            res_antiguos = supabase.table("licitaciones").select("id").eq("fuente", "Gobierno de La Rioja").eq("es_actualizada", True).limit(200).execute()
            if not res_antiguos.data:
                break
            ids_antiguos = [item["id"] for item in res_antiguos.data]
            for i in range(0, len(ids_antiguos), 50):
                lote_ids = ids_antiguos[i:i+50]
                supabase.table("licitaciones").update({
                    "es_novedad": False,
                    "es_actualizada": False
                }).in_("id", lote_ids).execute()
    except Exception as e:
        print(f"Aviso al resetear flags actualizados: {e}")

    # 2. Cargar registros existentes en Supabase para validar duplicados por TÍTULO y ÓRGANO
    try:
        existentes_resp = supabase.table("licitaciones").select("id, titulo, organo, fuente, tipo_contrato, fecha_fin, importe").execute()
        registros_db = {
            (str(item["titulo"]).strip().lower(), str(item["organo"]).strip().lower()): item 
            for item in existentes_resp.data if item.get("titulo") and item.get("organo")
        }
    except Exception as e:
        print(f"Error conectando con Supabase para lectura: {e}")
        return

    licitaciones_validas = []
    enlaces_procesados_sesion = set()
    estados_excluidos = {"resuelta", "cerrada", "adjudicada", "anulada", "desistida", "renunciada"}

    for aviso in results:
        # Filtrar estados cerrados / resueltos / adjudicados
        estado_licitacion = str(aviso.get("ESTADO_LICITACION", "")).strip().lower()
        if any(est in estado_licitacion for est in estados_excluidos):
            continue

        # Filtrar por fecha de publicación (últimos 2 días)
        fecha_pub_raw = aviso.get("FECHA_PUBLICA_LICITACION")
        if not fecha_pub_raw:
            continue
            
        try:
            f_pub_str = str(fecha_pub_raw)[:10].replace("/", "-")
            pub_date = datetime.strptime(f_pub_str, "%Y-%m-%d").date()
            if pub_date < limite_fecha:
                continue
        except Exception:
            continue

        enlace_raw = aviso.get("ENLACE_LICITACION") or aviso.get("ENLACE_PERFIL_CONTRATANTE")
        enlace = str(enlace_raw).strip() if enlace_raw else ""
        if not enlace:
            continue

        titulo_str = str(aviso.get("OBJETO_CONTRATO", "Sin título")).strip().strip('"')
        organo_str = str(aviso.get("ORGANO_CONTRATACION", "No especificado")).strip()
        fecha_pub = f_pub_str

        # Usar FECHA_HORA_FIN_OFERTAS para fecha_fin
        fecha_fin_str = "No especificada"
        fecha_fin_raw = aviso.get("FECHA_HORA_FIN_OFERTAS")
        if fecha_fin_raw:
            fecha_fin_str = str(fecha_fin_raw)[:10].replace("/", "-")

        importe_val = aviso.get("PRESUPUESTO_LICITACION") or aviso.get("VALOR_ESTIMADO") or 0.0
        try:
            importe = float(importe_val)
        except (ValueError, TypeError):
            importe = 0.0

        cpv_raw = aviso.get("CPVS", "")
        if cpv_raw:
            partes = [p.strip().split(',')[0] for p in str(cpv_raw).split(';') if p.strip()]
            cpv_limpios = list(dict.fromkeys([p for p in partes if p.isdigit()]))
            cpv = ", ".join(cpv_limpios) if cpv_limpios else "No especificado"
        else:
            cpv = "No especificado"

        lugar_bruto = aviso.get("LUGAR_EJECUCION", "ES230")
        lugar_ejecucion = procesar_lugar_rioja(lugar_bruto)

        tipo_contrato = str(aviso.get("TIPO_CONTRATO", "No especificado")).strip()

        if enlace in enlaces_procesados_sesion:
            continue
        enlaces_procesados_sesion.add(enlace)

        texto_completo = f"passage: Título: {titulo_str}. Órgano: {organo_str}. Tipo de contrato: {tipo_contrato}. CPV: {cpv}. Lugar: {lugar_ejecucion}. Importe: {importe} EUR."

        # Comprobación de duplicado por Título y Órgano (normalizados)
        clave_duplicado = (titulo_str.lower(), organo_str.lower())

        if clave_duplicado in registros_db:
            reg_antiguo = registros_db[clave_duplicado]
            fuente_actual = reg_antiguo.get("fuente", "")
            tipo_actual = reg_antiguo.get("tipo_contrato", "")
            
            actualizar_datos = {}
            
            if "Gobierno de La Rioja" not in fuente_actual:
                nueva_fuente = f"{fuente_actual}, Gobierno de La Rioja" if fuente_actual else "Gobierno de La Rioja"
                actualizar_datos["fuente"] = nueva_fuente
                actualizar_datos["es_actualizada"] = True

            if (not tipo_actual or tipo_actual == "No especificado") and tipo_contrato != "No especificado":
                actualizar_datos["tipo_contrato"] = tipo_contrato
                actualizar_datos["es_actualizada"] = True

            if reg_antiguo.get("importe") != importe:
                actualizar_datos["importe"] = importe
                actualizar_datos["es_actualizada"] = True

            if actualizar_datos:
                try:
                    supabase.table("licitaciones").update(actualizar_datos).eq("id", reg_antiguo["id"]).execute()
                except Exception as e:
                    print(f"Error actualizando registro existente por título/órgano: {e}")
            continue

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
            "fuente": "Gobierno de La Rioja"
        }

        licitaciones_validas.append(elemento)

    # 4. Inserción optimizada con lotes y reintentos
    if licitaciones_validas:
        total_a_subir = len(licitaciones_validas)
        print(f"Subiendo un total de {total_a_subir} licitaciones recientes a Supabase...")
        
        tamano_lote = 5
        subidas_exitosas = 0
        max_intentos = 3
        
        for i in range(0, total_a_subir, tamano_lote):
            lote = licitaciones_validas[i:i + tamano_lote]
            num_lote = i // tamano_lote + 1
            
            for intento in range(1, max_intentos + 1):
                try:
                    supabase.table("licitaciones").upsert(lote, on_conflict="enlace").execute()
                    subidas_exitosas += len(lote)
                    print(f"Progreso: {subidas_exitosas}/{total_a_subir} licitaciones procesadas...")
                    break
                except Exception as e:
                    print(f"⚠️ Intento {intento}/{max_intentos} fallido para lote Rioja {num_lote}: {e}")
                    if intento < max_intentos:
                        time.sleep(2 * intento)
                    else:
                        print(f"❌ Error definitivo al subir lote Rioja {num_lote}.")
        
        print(f"Sincronización completada con éxito. Se han subido/actualizado {subidas_exitosas} de {total_a_subir} licitaciones.")
    else:
        print("No hay licitaciones nuevas publicadas en los últimos 2 días para procesar.")

if __name__ == "__main__":
    sincronizar_licitaciones_rioja()
