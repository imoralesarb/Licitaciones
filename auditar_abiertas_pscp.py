from datetime import datetime, date
import os
import time
from sodapy import Socrata
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

# CONFIGURACIÓN

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small) para auditoría PSCP Catalunya...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

TAMANO_LOTE = 10  # Lotes de 10 en 10 para agilizar y evitar bloqueos

def procesar_lugar(lugar_raw):
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"
    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "Cataluña"
    if "cataluña" not in lugar_limpio.lower() and "catalunya" not in lugar_limpio.lower():
        return f"{lugar_limpio}, Cataluña"
    return lugar_limpio

def auditar_licitaciones_abiertas_pscp():
    hoy = date.today()
    dataset_id = "ybgg-dgi6"
    client = Socrata("analisi.transparenciacatalunya.cat", None)

    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    lote_contador = 1
    ids_procesados = []  # Control para evitar repetir registros en bucle

    print("Iniciando auditoría por lotes de licitaciones PSCP Catalunya (Fecha 'No especificada')...\n")

    while True:
        try:
            query = (
                supabase.table("licitaciones")
                .select("*")
                .eq("fuente", "PSCP Catalunya")
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
            print("\n¡Proceso finalizado! No quedan más licitaciones de PSCP Catalunya pendientes de auditar.")
            break

        print(f"\n --- Procesando Lote PSCP {lote_contador} ({len(registros)} registros) ---")
        ids_a_borrar = []

        for reg in registros:
            rec_id = reg.get("id")
            ids_procesados.append(rec_id)  # Registramos el ID para no volver a pedirlo
            
            enlace = reg.get("enlace", "")
            titulo = reg.get("titulo", "Sin título")

            if not enlace:
                continue

            # Consultar en la API de Socrata (PSCP) filtrando por la URL o identificador del enlace
            # Como el enlace suele estar dentro de un campo complejo o URL, filtramos por coincidencia de texto
            query_socrata = f"enllac_publicacio like '%{enlace}%'"

            try:
                chunk = client.get(
                    dataset_id,
                    where=query_socrata,
                    limit=1
                )
                
                if not chunk:
                    ids_a_borrar.append(rec_id)
                    print(f"   [A BORRAR - Ya no está en PSCP]: {titulo[:50]}...")
                    total_eliminadas += 1
                    continue

                aviso = chunk[0]
                fase = str(aviso.get("fase_publicacio", "")).lower()

                # Comprobar si ha cambiado de fase (ej. adjudicado, formalizado, etc.)
                if "anunci de licitació" not in fase and fase != "":
                    ids_a_borrar.append(rec_id)
                    print(f"    [A BORRAR - Cambio de fase a '{fase}']: {titulo[:50]}...")
                    total_eliminadas += 1
                    continue

                # Comprobar si ahora ya tiene fecha fin (termini_presentacio_ofertes)
                fecha_cierre_raw = aviso.get("termini_presentacio_ofertes")
                nueva_fecha_fin = "No especificada"
                if fecha_cierre_raw:
                    nueva_fecha_fin = fecha_cierre_raw[:10]

                # Verificar si ha caducado con la nueva fecha encontrada
                if nueva_fecha_fin != "No especificada":
                    try:
                        f_fin_date = datetime.strptime(nueva_fecha_fin, "%Y-%m-%d").date()
                        if f_fin_date < hoy:
                            ids_a_borrar.append(rec_id)
                            print(f"   [A BORRAR - Caducada]: {titulo[:50]}...")
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
                    print(f"    [ACTUALIZADA Fecha Fin PSCP a {nueva_fecha_fin}]: {titulo[:50]}...")
                    total_actualizadas += 1
                else:
                    total_sin_cambios += 1

                time.sleep(0.2)

            except Exception as e:
                print(f"    Error procesando registro PSCP {enlace}: {e}")
                continue

        # Borrado en bloque por lotes usando los IDs recolectados
        if ids_a_borrar:
            try:
                supabase.table("licitaciones").delete().in_("id", ids_a_borrar).execute()
                print(f"    -> ¡{len(ids_a_borrar)} licitaciones PSCP eliminadas de Supabase en este lote!")
            except Exception as e:
                print(f"    Error al eliminar lote en Supabase: {e}")

        lote_contador += 1
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print(" RESUMEN FINAL AUDITORÍA PSCP CATALUNYA:")
    print(f"  - Eliminadas (adjudicadas/cambio de fase/caducadas): {total_eliminadas}")
    print(f"  - Actualizadas (con nueva fecha): {total_actualizadas}")
    print(f"  - Sin cambios: {total_sin_cambios}")
    print(" ¡Auditoría de licitaciones PSCP Catalunya completada con éxito!")

if __name__ == "__main__":
    auditar_licitaciones_abiertas_pscp()
