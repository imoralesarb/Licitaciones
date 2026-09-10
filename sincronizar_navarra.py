# -*- coding: utf-8 -*-
from datetime import datetime, date, timedelta
import os
import time
import re
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import requests

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small)...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")


def limpiar_cpv(cpv_raw):
    """Extrae los primeros 8 dígitos del CPV."""
    if not cpv_raw or cpv_raw == "No especificado":
        return "No especificado"

    cpv_str = str(cpv_raw).strip()
    match = re.search(r"(\d{8}(?:-\d)?)", cpv_str)

    if match:
        return match.group(1)

    return cpv_str[:20]


def procesar_lugar_navarra(lugar_raw):
    """Añade Navarra al lugar de ejecución si no lo especifica."""
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"

    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "Navarra"

    if "navarra" not in lugar_limpio.lower():
        return f"{lugar_limpio}, Navarra"

    return lugar_limpio


def sincronizar_licitaciones_navarra():
    hoy_date = datetime.now().date()
    limite_fecha = hoy_date - timedelta(days=2)

    print(f"Filtrando licitaciones publicadas desde {limite_fecha} hasta {hoy_date}")

    resource_id = "dda1af7c-0dcd-4992-9852-ded6b1e7625d"
    url_api = f"https://datosabiertos.navarra.es/es/api/3/action/datastore_search?resource_id={resource_id}&limit=1000"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    # ========================================================
    # 1. DESCARGAR DATOS DE NAVARRA
    # ========================================================
    print("Consultando la API de Contratación de Navarra...")

    try:
        response = requests.get(url_api, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"Error HTTP {response.status_code}")
            return

        data = response.json()
        results = data.get("result", {}).get("records", [])

    except Exception as e:
        print(f"Error conectando con la API de Navarra: {e}")
        return

    print(f"Total registros obtenidos de la API Navarra: {len(results)}")

    # ========================================================
    # 2. CARGAR REGISTROS EXISTENTES DE SUPABASE
    # ========================================================
    try:
        existentes_resp = (
            supabase.table("licitaciones")
            .select(
                "id, enlace, titulo, organo, fuente, fecha, importe, tipo_contrato, cpv, fecha_fin, es_novedad, es_actualizada"
            )
            .execute()
        )

        registros_db = {}
        ids_flags_navarra = []

        for item in existentes_resp.data:
            enlace_item = item.get("enlace")
            if enlace_item:
                registros_db[enlace_item] = item

            fuente_item = str(item.get("fuente", ""))
            if "contratación navarra" in fuente_item.lower() and (
                item.get("es_novedad") is True or item.get("es_actualizada") is True
            ):
                ids_flags_navarra.append(item["id"])

        print(f"Registros cargados desde Supabase para validación: {len(existentes_resp.data)}")

    except Exception as e:
        print(f"Error conectando con Supabase para lectura: {e}")
        return

    # ========================================================
    # 2.1. RESETEAR ETIQUETAS ANTERIORES DE NAVARRA
    # ========================================================
    if ids_flags_navarra:
        print(f"Reseteando etiquetas anteriores de {len(ids_flags_navarra)} registros de Navarra...")

        tamano_reset = 25
        max_intentos_reset = 3
        reset_correcto = True
        total_reseteadas = 0

        for i in range(0, len(ids_flags_navarra), tamano_reset):
            lote_ids = ids_flags_navarra[i:i + tamano_reset]
            num_lote_reset = (i // tamano_reset) + 1
            exito_lote = False

            for intento in range(1, max_intentos_reset + 1):
                try:
                    supabase.table("licitaciones").update({
                        "es_novedad": False,
                        "es_actualizada": False
                    }).in_("id", lote_ids).execute()

                    total_reseteadas += len(lote_ids)
                    print(f"  -> Lote de etiquetas {num_lote_reset} reseteado con éxito ({len(lote_ids)} registros).")
                    exito_lote = True
                    break
                except Exception as e:
                    print(f"  -> Intento {intento}/{max_intentos_reset} fallido para lote de etiquetas {num_lote_reset}: {e}")
                    if intento < max_intentos_reset:
                        time.sleep(2 * intento)
                    else:
                        print(f"  -> Error definitivo al resetear el lote de etiquetas {num_lote_reset}.")
                        reset_correcto = False

            if not exito_lote:
                continue

        if reset_correcto:
            print(f"Etiquetas anteriores reseteadas correctamente: {total_reseteadas} registros.")
        else:
            print("Aviso: no se pudieron resetear todas las etiquetas anteriores.")
    else:
        print("No hay etiquetas anteriores de Navarra que resetear.")

    # ========================================================
    # 3. PROCESAR LICITACIONES
    # ========================================================
    licitaciones_validas = []
    enlaces_procesados_sesion = set()

    for i, aviso in enumerate(results, 1):
        fecha_pub_str = aviso.get("FechaPublicacion", "")
        if not fecha_pub_str:
            continue

        try:
            dt_pub = datetime.strptime(fecha_pub_str[:10], "%d/%m/%Y")
            pub_date = dt_pub.date()

            if pub_date < limite_fecha or pub_date > hoy_date:
                continue

            fecha_pub = dt_pub.strftime("%Y-%m-%d")
        except ValueError:
            continue

        ckan_id = aviso.get("_id", i)
        enlace = f"https://hacienda.navarra.es/sicpportal/mtoBuscadorAnuncios.aspx#{ckan_id}"

        if not enlace or enlace in enlaces_procesados_sesion:
            continue

        enlaces_procesados_sesion.add(enlace)

        titulo = str(aviso.get("BreveDescripcion") or "Sin descripción").strip()
        organo = str(aviso.get("Organo") or "No especificado").strip()
        entidad = str(aviso.get("Entidad") or "No especificada").strip()
        organo_completo = f"{entidad} - {organo}" if entidad else organo

        fecha_fin_str = "No especificada"

        importe_val = aviso.get("PrecioLicitacion") or aviso.get("ValorEstimado")
        try:
            val_limpio = (
                str(importe_val)
                .replace("€", "")
                .replace("EUR", "")
                .replace(".", "")
                .replace(",", ".")
                .strip()
            )
            importe = float(val_limpio) if val_limpio else 0.0
        except (ValueError, TypeError):
            importe = 0.0

        cpv = limpiar_cpv(aviso.get("CPV", "No especificado"))
        lugar_bruto = aviso.get("LugarEjecucion", "Navarra")
        lugar_ejecucion = procesar_lugar_navarra(lugar_bruto)
        tipo_contrato = str(aviso.get("TipoContrato", "No especificado")).capitalize()

        texto_completo = (
            f"passage: Título: {titulo}. Órgano: {organo_completo}. "
            f"Tipo Contrato: {tipo_contrato}. Lugar: {lugar_ejecucion}. "
            f"Importe: {importe} EUR. CPV: {cpv}."
        )

        if enlace in registros_db:
            reg_antiguo = registros_db[enlace]
            fuente_actual = reg_antiguo.get("fuente", "")
            tipo_actual = reg_antiguo.get("tipo_contrato", "")
            actualizar_datos = {}

            if "Contratación Navarra" not in fuente_actual:
                nueva_fuente = f"{fuente_actual}, Contratación Navarra" if fuente_actual else "Contratación Navarra"
                actualizar_datos["fuente"] = nueva_fuente

            if (not tipo_actual or tipo_actual == "No especificado") and tipo_contrato != "No especificado":
                actualizar_datos["tipo_contrato"] = tipo_contrato

            es_actualizado = (
                reg_antiguo.get("titulo") != titulo or
                reg_antiguo.get("importe") != importe
            )

            if es_actualizado:
                actualizar_datos["es_actualizada"] = True

            if actualizar_datos:
                try:
                    supabase.table("licitaciones").update(actualizar_datos).eq("enlace", enlace).execute()
                except Exception as e:
                    print(f"Error actualizando registro existente {enlace}: {e}")

            continue

        embedding = encoder.encode(texto_completo).tolist()

        elemento = {
            "titulo": titulo,
            "organo": organo_completo,
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
            "fuente": "Contratación Navarra",
        }

        licitaciones_validas.append(elemento)

    # ========================================================
    # 4. INSERTAR / ACTUALIZAR EN SUPABASE
    # ========================================================
    if licitaciones_validas:
        total_a_subir = len(licitaciones_validas)
        print(f"Subiendo un total de {total_a_subir} licitaciones publicadas entre {limite_fecha} y {hoy_date} a Supabase...")

        tamano_lote = 5
        subidas_exitosas = 0
        max_intentos = 3

        for i in range(0, total_a_subir, tamano_lote):
            lote = licitaciones_validas[i:i + tamano_lote]
            num_lote = (i // tamano_lote) + 1

            for intento in range(1, max_intentos + 1):
                try:
                    supabase.table("licitaciones").upsert(lote, on_conflict="enlace").execute()
                    subidas_exitosas += len(lote)
                    print(f"Progreso: {subidas_exitosas}/{total_a_subir} licitaciones procesadas...")
                    break
                except Exception as e:
                    print(f"⚠️ Intento {intento}/{max_intentos} fallido para lote Navarra {num_lote}: {e}")
                    if intento < max_intentos:
                        time.sleep(2 * intento)
                    else:
                        print(f"❌ Error definitivo al subir lote Navarra {num_lote}.")

        print(f"Sincronización completada con éxito. Se han subido/actualizado {subidas_exitosas} de {total_a_subir} licitaciones.")
    else:
        print(f"No hay licitaciones publicadas entre {limite_fecha} y {hoy_date} para procesar.")


if __name__ == "__main__":
    sincronizar_licitaciones_navarra()
