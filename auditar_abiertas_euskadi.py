# -*- coding: utf-8 -*-
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

print("Cargando modelo de IA (multilingual-e5-small) para auditoría Euskadi...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

TAMANO_LOTE = 10  # Lotes de 10 en 10 para agilizar y evitar bloqueos


# ============================================================
# FUNCIONES DE AUDITORÍA
# ============================================================

def auditar_licitaciones_abiertas_euskadi():
    hoy = date.today()
    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    lote_contador = 1
    ids_procesados = []  # Control para evitar repetir registros en bucle

    print("Iniciando auditoría por lotes de licitaciones Euskadi (Auditando fechas y disponibilidad en API)...\n")

    while True:
        try:
            query = (
                supabase.table("licitaciones")
                .select("*")
                .eq("fuente", "Euskadi")
                .eq("fecha_fin", "No especificada")
                .limit(TAMANO_LOTE)
            )
            
            # Excluimos los IDs ya procesados en esta ejecución
            if ids_procesados:
                query = query.not_.in_("id", ids_procesados)

            response = query.execute()
            registros = response.data
        except Exception as e:
            print(f"Error al consultar Supabase: {e}")
            break

        if not registros:
            print("\n¡Proceso finalizado! No quedan más licitaciones de Euskadi pendientes de auditar.")
            break

        print(f"\n --- Procesando Lote Euskadi {lote_contador} ({len(registros)} registros) ---")
        ids_a_borrar = []

        for reg in registros:
            rec_id = reg.get("id")
            ids_procesados.append(rec_id)  # Registramos el ID para no volver a pedirlo
            
            enlace = reg.get("enlace", "")
            titulo = reg.get("titulo", "Sin título")

            if not enlace:
                continue

            # Extraemos el código de ítem/record desde el enlace de Euskadi
            codigo_item = None
            if "anuncio_contratacion/" in enlace:
                try:
                    partes = enlace.split("anuncio_contratacion/")
                    if len(partes) > 1:
                        codigo_item = partes[1].split("/")[0]
                except Exception:
                    pass

            if not codigo_item:
                ids_a_borrar.append(rec_id)
                print(f"    [A BORRAR - No se pudo extraer código de la URL]: {titulo[:50]}...")
                total_eliminadas += 1
                continue

            try:
                url_detalle = f"https://api.euskadi.eus/procurements/contracting-notices/{codigo_item}"
                resp_detalle = requests.get(url_detalle, headers={"Accept": "application/json"}, timeout=5)
                
                # Si la API da error (404 u otro), significa que el anuncio ya no está activo o se ha retirado/adjudicado
                if resp_detalle.status_code != 200:
                    ids_a_borrar.append(rec_id)
                    print(f"    [A BORRAR - Retirado o Adjudicado (API dio error {resp_detalle.status_code})]: {titulo[:50]}...")
                    total_eliminadas += 1
                    continue

                det_data = resp_detalle.json()

                # Comprobar si hay fecha de fin/cierre nueva en el detalle
                deadline_raw = det_data.get("endDate")
                nueva_fecha_fin = "No especificada"
                if deadline_raw:
                    nueva_fecha_fin = deadline_raw[:10]

                # Verificar si ha caducado con la nueva fecha encontrada
                if nueva_fecha_fin != "No especificada":
                    try:
                        f_fin_date = datetime.strptime(nueva_fecha_fin, "%Y-%m-%d").date()
                        if f_fin_date < hoy:
                            ids_a_borrar.append(rec_id)
                            print(f"    [A BORRAR - Caducada con fecha {nueva_fecha_fin}]: {titulo[:50]}...")
                            total_eliminadas += 1
                            continue
                    except ValueError:
                        pass

                # Si la fecha fin cambió de 'No especificada' a una fecha válida futura, la actualizamos
                if nueva_fecha_fin != reg.get("fecha_fin"):
                    reg["fecha_fin"] = nueva_fecha_fin
                    reg["es_actualizada"] = True
                    
                    texto_completo = reg.get("texto_completo", "")
                    reg["embedding"] = encoder.encode(texto_completo).tolist()

                    supabase.table("licitaciones").upsert(reg, on_conflict="enlace").execute()
                    print(f"    [ACTUALIZADA Fecha Fin Euskadi a {nueva_fecha_fin}]: {titulo[:50]}...")
                    total_actualizadas += 1
                else:
                    total_sin_cambios += 1

                time.sleep(0.2)

            except Exception as e:
                print(f"    Error procesando registro Euskadi {enlace}: {e}")
                continue

        # Borrado en bloque por lotes usando los IDs recolectados
        if ids_a_borrar:
            try:
                supabase.table("licitaciones").delete().in_("id", ids_a_borrar).execute()
                print(f"    -> ¡{len(ids_a_borrar)} licitaciones Euskadi eliminadas de Supabase en este lote!")
            except Exception as e:
                print(f"    Error al eliminar lote en Supabase: {e}")

        lote_contador += 1
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print(" RESUMEN FINAL AUDITORÍA EUSKADI:")
    print(f"  - Eliminadas (retiradas, adjudicadas o caducadas): {total_eliminadas}")
    print(f"  - Actualizadas (con nueva fecha): {total_actualizadas}")
    print(f"  - Sin cambios: {total_sin_cambios}")
    print(" ¡Auditoría de licitaciones Euskadi completada con éxito!")


if __name__ == "__main__":
    auditar_licitaciones_abiertas_euskadi()
