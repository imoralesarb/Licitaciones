from datetime import datetime, timedelta
import os
import time

from sodapy import Socrata
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
# TRADUCCIÓN DEL TIPO DE CONTRATO
# ============================================================
def traducir_tipo_contrato(tipus_cat):
    """
    Traduce los tipos de contrato de la PSCP
    al castellano para la BBDD.
    """
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

# ============================================================
# PROCESAR LUGAR DE EJECUCIÓN
# ============================================================
def procesar_lugar(lugar_raw):
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"
    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "Cataluña"
    if "cataluña" not in lugar_limpio.lower() and "catalunya" not in lugar_limpio.lower():
        return f"{lugar_limpio}, Cataluña"
    return lugar_limpio

# ============================================================
# QUITAR PSCP DE UNA FUENTE COMBINADA
# ============================================================
def quitar_fuente_pscp(fuente):
    """
    Elimina 'PSCP Catalunya' de una cadena de fuentes.
    """
    if not fuente:
        return ""
    fuentes = [f.strip() for f in str(fuente).split(",") if f.strip()]
    fuentes_restantes = [
        f for f in fuentes
        if f.casefold() != "pscp catalunya".casefold()
    ]
    return ", ".join(fuentes_restantes)

# ============================================================
# SINCRONIZACIÓN PSCP
# ============================================================
def sincronizar_licitaciones_pscp():
    hoy_date = datetime.now().date()
    ayer_date = hoy_date - timedelta(days=1)
    fecha_inicio = ayer_date.strftime("%Y-%m-%dT00:00:00")
    fecha_fin = (hoy_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    dataset_id = "ybgg-dgi6"

    client = Socrata("analisi.transparenciacatalunya.cat", None)
    query = (
        f"data_publicacio_anunci >= '{fecha_inicio}' "
        f"AND data_publicacio_anunci < '{fecha_fin}' "
        f"AND fase_publicacio = 'Anunci de licitació'"
    )

    print("Consultando la API de la PSCP...")

    # ========================================================
    # 1. OBTENER DATOS DE LA API
    # ========================================================
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

    # ========================================================
    # 2. CARGAR TODOS LOS REGISTROS EXISTENTES
    # ========================================================
    try:
        existentes_resp = (
            supabase
            .table("licitaciones")
            .select(
                "id, enlace, titulo, organo, fuente, "
                "importe, tipo_contrato, fecha_fin, "
                "es_novedad, es_actualizada"
            )
            .execute()
        )
        registros_db = {
            item["enlace"]: item
            for item in existentes_resp.data
            if item.get("enlace")
        }
    except Exception as e:
        print(f"Error conectando con Supabase para lectura: {e}")
        return

    # ========================================================
    # 3. RESETEAR FLAGS DE PSCP
    # ========================================================
    print("Reseteando flags de novedades y actualizaciones anteriores de PSCP...")
    ids_flags_pscp = []

    for item in existentes_resp.data:
        fuente_item = str(item.get("fuente", ""))
        if (
            "pscp catalunya" in fuente_item.casefold()
            and (item.get("es_novedad") is True or item.get("es_actualizada") is True)
        ):
            ids_flags_pscp.append(item["id"])

    print(f"Registros de PSCP con flags activadas: {len(ids_flags_pscp)}")
    tamano_reset = 25
    max_intentos_reset = 3
    total_reseteadas = 0

    for i in range(0, len(ids_flags_pscp), tamano_reset):
        lote_ids = ids_flags_pscp[i:i + tamano_reset]
        reset_correcto = False

        for intento in range(1, max_intentos_reset + 1):
            try:
                supabase.table("licitaciones").update({
                    "es_novedad": False,
                    "es_actualizada": False
                }).in_("id", lote_ids).execute()
                total_reseteadas += len(lote_ids)
                reset_correcto = True
                break
            except Exception as e:
                print(f"⚠️ Error reseteando lote {i // tamano_reset + 1} (intento {intento}/{max_intentos_reset}): {e}")
                if intento < max_intentos_reset:
                    time.sleep(2 * intento)

        if not reset_correcto:
            print(f"❌ No se pudo resetear el lote {i // tamano_reset + 1}.")

    print(f"Flags reseteados: {total_reseteadas}")

    # ========================================================
    # 4. PROCESAR REGISTROS DE LA API
    # ========================================================
    licitaciones_validas = []
    filtrados_caducados = 0
    enlaces_procesados_sesion = set()

    for aviso in results:
        enlace_raw = aviso.get("enllac_publicacio")
        if isinstance(enlace_raw, dict) and "url" in enlace_raw:
            enlace = enlace_raw["url"]
        else:
            enlace = str(enlace_raw)

        if not enlace or enlace == "None":
            continue

        if enlace in enlaces_procesados_sesion:
            continue
        enlaces_procesados_sesion.add(enlace)

        titulo_str = str(aviso.get("denominacio", "Sin título")).strip()
        organo_str = str(aviso.get("nom_organ", "No especificado")).strip()

        fecha_fin_str = "No especificada"
        fecha_cierre_raw = aviso.get("termini_presentacio_ofertes")
        if fecha_cierre_raw:
            fecha_fin_str = str(fecha_cierre_raw)[:10]
            try:
                cierre_date = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                if cierre_date < hoy_date:
                    filtrados_caducados += 1
                    continue
            except ValueError:
                pass

        fecha_pub = str(aviso.get("data_publicacio_anunci", ""))[:10]

        importe_val = (
            aviso.get("pressupost_licitacio_sense_iva_expedient")
            or aviso.get("pressupost_licitacio_sense_1")
            or 0.0
        )
        try:
            importe = float(importe_val)
        except (ValueError, TypeError):
            importe = 0.0

        cpv_raw = aviso.get("codi_cpv", "No especificado")
        if cpv_raw and cpv_raw != "No especificado":
            cpv = ", ".join([c.strip() for c in str(cpv_raw).split("||") if c.strip()])
        else:
            cpv = "No especificado"

        lugar_bruto = aviso.get("lloc_execucio", "No especificado")
        lugar_ejecucion = procesar_lugar(lugar_bruto)

        tipus_cat = aviso.get("tipus_contracte", "No especificado")
        tipo_contrato = traducir_tipo_contrato(tipus_cat)

        texto_completo = (
            f"passage: Título: {titulo_str}. Órgano: {organo_str}. "
            f"Tipo de contrato: {tipo_contrato}. CPV: {cpv}. "
            f"Lugar: {lugar_ejecucion}. Importe: {importe} EUR."
        )

        # ====================================================
        # 5. REGISTRO EXISTENTE
        # ====================================================
        if enlace in registros_db:
            reg_antiguo = registros_db[enlace]
            fuente_actual = str(reg_antiguo.get("fuente", ""))
            tipo_actual = str(reg_antiguo.get("tipo_contrato", ""))
            actualizar_datos = {}

            if "pscp catalunya" not in fuente_actual.casefold():
                if fuente_actual:
                    nueva_fuente = f"{fuente_actual}, PSCP Catalunya"
                else:
                    nueva_fuente = "PSCP Catalunya"
                actualizar_datos["fuente"] = nueva_fuente

            if (not tipo_actual or tipo_actual == "No especificado") and tipo_contrato != "No especificado":
                actualizar_datos["tipo_contrato"] = tipo_contrato

            es_actualizado = (
                reg_antiguo.get("titulo") != titulo_str
                or reg_antiguo.get("importe") != importe
                or reg_antiguo.get("fecha_fin") != fecha_fin_str
            )
            
            if es_actualizado:
                actualizar_datos["es_actualizada"] = True

            if actualizar_datos:
                try:
                    supabase.table("licitaciones").update(actualizar_datos).eq("enlace", enlace).execute()
                except Exception as e:
                    print(f"Error actualizando registro existente {enlace}: {e}")
            continue

        # ====================================================
        # 6. REGISTRO NUEVO
        # ====================================================
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

    print(f"Licitaciones descartadas por estar caducadas en la API: {filtrados_caducados}")

    # ========================================================
    # 7. LIMPIEZA DE CADUCADAS
    # ========================================================
    try:
        todos_db = (
            supabase
            .table("licitaciones")
            .select("id, enlace, fecha_fin, fuente")
            .ilike("fuente", "%PSCP Catalunya%")
            .execute()
        )
        ids_a_borrar = []
        ids_fuente_a_actualizar = []

        for item in todos_db.data:
            f_fin = item.get("fecha_fin")
            if not f_fin or f_fin == "No especificada":
                continue
            try:
                f_cierre = datetime.strptime(f_fin, "%Y-%m-%d").date()
                if f_cierre < hoy_date:
                    fuente_actual = str(item.get("fuente", ""))
                    nueva_fuente = quitar_fuente_pscp(fuente_actual)
                    if not nueva_fuente:
                        ids_a_borrar.append(item["id"])
                    else:
                        ids_fuente_a_actualizar.append((item["id"], nueva_fuente))
            except ValueError:
                pass

        if ids_a_borrar:
            for i in range(0, len(ids_a_borrar), 50):
                lote_ids = ids_a_borrar[i:i + 50]
                supabase.table("licitaciones").delete().in_("id", lote_ids).execute()
            print(f"Eliminadas {len(ids_a_borrar)} licitaciones caducadas de PSCP.")

        if ids_fuente_a_actualizar:
            total_fuentes_actualizadas = 0
            for rec_id, nueva_fuente in ids_fuente_a_actualizar:
                try:
                    supabase.table("licitaciones").update({"fuente": nueva_fuente}).eq("id", rec_id).execute()
                    total_fuentes_actualizadas += 1
                except Exception as e:
                    print(f"Error actualizando fuente del registro {rec_id}: {e}")
            print(f"PSCP eliminado de {total_fuentes_actualizadas} fuentes combinadas por caducidad.")
    except Exception as e:
        print(f"Error en la limpieza de caducadas: {e}")

    # ========================================================
    # 8. INSERCIÓN OPTIMIZADA
    # ========================================================
    if licitaciones_validas:
        total_a_subir = len(licitaciones_validas)
        print(f"Subiendo un total de {total_a_subir} licitaciones a Supabase...")
        tamano_lote = 5
        subidas_exitosas = 0
        max_intentos = 3

        for i in range(0, total_a_subir, tamano_lote):
            lote = licitaciones_validas[i:i + tamano_lote]
            num_lote = (i // tamano_lote) + 1
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

        print(f"Sincronización completada. Se han subido/actualizado {subidas_exitosas} de {total_a_subir} licitaciones.")
    else:
        print("No hay licitaciones nuevas para procesar en este rango.")

if __name__ == "__main__":
    sincronizar_licitaciones_pscp()
