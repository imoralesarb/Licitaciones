from datetime import date, datetime, timedelta
import os
import re
import lxml.etree as ET
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client
import time
import requests

# ============================================================
# 1. CONFIGURACIÓN Y MODELO DE IA
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small)...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

URL_ATOM_MADRID = "https://contratos-publicos.comunidad.madrid/feed/licitaciones2"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cac": "urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2",
    "cac-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2",
    "cbc-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2",
}

MAPEO_NUTS = {
    "ES11": "Galicia",
    "ES12": "Principado de Asturias",
    "ES13": "Cantabria",
    "ES21": "País Vasco",
    "ES22": "Comunidad Foral de Navarra",
    "ES23": "La Rioja",
    "ES24": "Aragón",
    "ES30": "Comunidad de Madrid",
    "ES41": "Castilla y León",
    "ES42": "Castilla-La Mancha",
    "ES43": "Extremadura",
    "ES51": "Cataluña",
    "ES52": "Comunidad Valenciana",
    "ES53": "Illes Balears",
    "ES61": "Andalucía",
    "ES62": "Región de Murcia",
    "ES63": "Ciudad Autónoma de Ceuta",
    "ES64": "Ciudad Autónoma de Melilla",
    "ES70": "Canarias",
    "ES300": "Madrid"
}

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def normalizar_organo(texto):
    if not texto:
        return ""
    texto = texto.lower().strip()
    texto = re.sub(r'[áàäâ]', 'a', texto)
    texto = re.sub(r'[éèëê]', 'e', texto)
    texto = re.sub(r'[íìïî]', 'i', texto)
    texto = re.sub(r'[óòöô]', 'o', texto)
    texto = re.sub(r'[úùüû]', 'u', texto)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return re.sub(r'\s+', ' ', texto)

def _texto(el, xpath, ns=NS):
    nodo = el.find(xpath, ns)
    return nodo.text.strip() if nodo is not None and nodo.text else None

def traducir_tipo_contrato_madrid(codigo_raw):
    """Traduce o mapea el código o texto del tipo de contrato de la Comunidad de Madrid (PLACE) al estándar."""
    if not codigo_raw:
        return "No especificado"
    limpio = str(codigo_raw).strip().lower()
    mapping_codigos = {
        "1": "Suministros",
        "2": "Servicios",
        "3": "Obras",
        "4": "Concesión de obras",
        "5": "Gestión de servicios públicos",
        "6": "Concesión de servicios",
        "7": "Colaboración entre el sector público y el sector privado",
        "8": "Administrativo especial",
        "21": "Privado",
        "patrimonial": "Patrimonial",
        "otros": "Otros"
    }
    if limpio in mapping_codigos:
        return mapping_codigos[limpio]
    mapping_texto = {
        "obres": "Obras",
        "obras": "Obras",
        "serveis": "Servicios",
        "servicios": "Servicios",
        "subministraments": "Suministros",
        "suministros": "Suministros",
        "concesión de obras públicas": "Concesión de obras públicas",
        "concesión de obras": "Concesión de obras",
        "gestión de servicios públicos": "Gestión de servicios públicos",
        "concesión de servicios": "Concesión de servicios",
        "colaboración entre el sector público y el sector privado": "Colaboración entre el sector público y el sector privado",
        "administrativo especial": "Administrativo especial",
        "privado": "Privado",
        "patrimonial": "Patrimonial",
        "otros": "Otros"
    }
    return mapping_texto.get(limpio, str(codigo_raw).capitalize())

# ============================================================
# 3. SINCRONIZACIÓN COMUNIDAD DE MADRID
# ============================================================

def sincronizar_licitaciones_madrid():
    hoy = datetime.now().date()
    ayer = hoy - timedelta(days=2)
    fecha_inicio_rango = ayer
    fecha_fin_rango = hoy

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url_actual = URL_ATOM_MADRID
    entries_totales = []
    max_paginas = 10
    paginas_procesadas = 0

    print("Descargando datos del feed ATOM de la Comunidad de Madrid...")

    while url_actual and paginas_procesadas < max_paginas:
        paginas_procesadas += 1
        print(f"  -> Descargando página {paginas_procesadas} (URL: {url_actual})")
        try:
            resp = requests.get(url_actual, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"  ⚠️ Error HTTP {resp.status_code} al descargar la página.")
                break
            parser = ET.XMLParser(recover=True)
            root = ET.fromstring(resp.content, parser=parser)
            entries = root.findall("atom:entry", NS)
            if not entries:
                entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if not entries:
                print("  ℹ️ No se encontraron más entradas en esta página.")
                break
            print(f"     + {len(entries)} entradas encontradas en esta página.")
            entries_totales.extend(entries)
            next_link_el = root.find("atom:link[@rel='next']", NS)
            if next_link_el is None:
                next_link_el = root.find(".//{http://www.w3.org/2005/Atom}link[@rel='next']")
            url_actual = next_link_el.get("href") if next_link_el is not None else None
            time.sleep(0.5)
        except Exception as e:
            print(f"  ❌ Error descargando feed: {e}")
            break

    print(f"\nTotal de entradas acumuladas en el feed: {len(entries_totales)}")

    # ========================================================
    # 1. CARGAR REGISTROS EXISTENTES DE SUPABASE
    # ========================================================
    print("Cargando registros existentes desde Supabase para validación global...")
    try:
        existentes_resp = (
            supabase.table("licitaciones")
            .select("id, enlace, titulo, organo, fuente, fecha, importe, tipo_contrato, cpv, fecha_fin, es_novedad, es_actualizada")
            .execute()
        )
        registros_db = {}
        registros_existentes = set()
        ids_flags_madrid = []

        for item in existentes_resp.data:
            enlace_item = item.get("enlace")
            if enlace_item:
                registros_db[enlace_item] = item
            t = str(item.get("titulo", "")).strip().lower()
            o_base = normalizar_organo(item.get("organo", ""))
            if t or o_base:
                registros_existentes.add((t, o_base))
            fuente_item = str(item.get("fuente", ""))
            if "comunidad de madrid" in fuente_item.lower() and (item.get("es_novedad") is True or item.get("es_actualizada") is True):
                ids_flags_madrid.append(item["id"])

        print(f"Registros totales cargados desde Supabase: {len(registros_db)}")
    except Exception as e:
        print(f"Error conectando con Supabase para lectura: {e}")
        return

    # ========================================================
    # 1.1. RESETEAR ETIQUETAS ANTERIORES DE MADRID
    # ========================================================
    if ids_flags_madrid:
        print(f"Reseteando etiquetas anteriores de {len(ids_flags_madrid)} registros de Madrid...")
        tamano_reset = 25
        max_intentos_reset = 3
        reset_correcto = True
        total_reseteadas = 0

        for i in range(0, len(ids_flags_madrid), tamano_reset):
            lote_ids = ids_flags_madrid[i:i + tamano_reset]
            num_lote_reset = (i // tamano_reset) + 1
            exito_lote = False

            for intento in range(1, max_intentos_reset + 1):
                try:
                    supabase.table("licitaciones").update({"es_novedad": False, "es_actualizada": False}).in_("id", lote_ids).execute()
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
        print("No hay etiquetas anteriores de Madrid que resetear.")

    # ========================================================
    # 2. PROCESAR LICITACIONES
    # ========================================================
    licitaciones_validas = []
    filtrados_caducados = 0
    enlaces_procesados_sesion = set()

    print(f"Procesando y filtrando entradas (del {fecha_inicio_rango} al {fecha_fin_rango})...")

    for entry in entries_totales:
        enlace_el = entry.find("atom:link", NS)
        enlace = enlace_el.get("href") if enlace_el is not None else ""
        enlace = enlace.strip()
        if not enlace:
            continue

        txt_updated = _texto(entry, "atom:updated")
        txt_published = _texto(entry, "atom:published")
        txt_fecha = txt_updated or txt_published
        if not txt_fecha:
            continue

        fecha_pub = txt_fecha.split("T")[0]
        try:
            date_pub_obj = datetime.strptime(fecha_pub, "%Y-%m-%d").date()
        except ValueError:
            continue

        if not (fecha_inicio_rango <= date_pub_obj <= fecha_fin_rango):
            continue

        end_date_el = entry.find(".//cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod/cbc:EndDate", NS)
        fecha_fin_str = "No especificada"

        if end_date_el is not None and end_date_el.text:
            fecha_fin_str = end_date_el.text.strip()[:10]
            try:
                cierre_date = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                if cierre_date < hoy:
                    filtrados_caducados += 1
                    continue
            except ValueError:
                pass

        titulo_str = _texto(entry, "atom:title") or "Sin título"
        organo = "Órgano desconocido"
        rutas_organo = [
            ".//cac-place-ext:LocatedContractingParty//cac:PartyName//cbc:Name",
            ".//cac:ContractingParty//cac:PartyName//cbc:Name",
            ".//cbc:PartyName//cbc:Name"
        ]

        for ruta in rutas_organo:
            organo_el = entry.find(ruta, NS)
            if organo_el is not None and organo_el.text and organo_el.text.strip():
                organo = organo_el.text.strip()
                break

        organo_base = normalizar_organo(organo)
        cpv_codigo = "No especificado"
        cpv_elements = entry.findall(".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cac:RequiredCommodityClassification/cbc:ItemClassificationCode", NS)
        if not cpv_elements:
            cpv_elements = entry.findall(".//cbc:ItemClassificationCode", NS)
        if cpv_elements:
            cpv_codigo = ", ".join([el.text.strip() for el in cpv_elements if el.text])

        lugar_ejecucion = "Madrid"
        lugar_el = entry.find(".//cac:ProcurementProject/cac:RealizedLocation/cbc:CountrySubentity", NS)
        if lugar_el is not None and lugar_el.text:
            lugar_ejecucion = lugar_el.text.strip()
        else:
            lugar_el = entry.find(".//cac:ProcurementProject/cac:RealizedLocation/cbc:CountrySubentityCode", NS)
            if lugar_el is not None and lugar_el.text:
                lugar_ejecucion = MAPEO_NUTS.get(lugar_el.text.strip(), lugar_el.text.strip())

        type_code_el = entry.find(".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cbc:TypeCode", NS)
        if type_code_el is None:
            type_code_el = entry.find(".//cbc:TypeCode", NS)

        tipo_contrato_raw = type_code_el.text.strip() if (type_code_el is not None and type_code_el.text) else "No especificado"
        tipo_contrato = traducir_tipo_contrato_madrid(tipo_contrato_raw)

        importe = 0.0
        presupuesto_el = entry.find(".//cac:BudgetAmount/cbc:EstimatedOverallContractAmount", NS)
        if presupuesto_el is None:
            presupuesto_el = entry.find(".//cac:BudgetAmount/cbc:TaxExclusiveAmount", NS)
        if presupuesto_el is None:
            presupuesto_el = entry.find(".//cac:BudgetAmount/cbc:TotalAmount", NS)

        if presupuesto_el is not None and presupuesto_el.text:
            try:
                importe = float(presupuesto_el.text.strip().replace(",", "."))
            except Exception:
                pass

        if enlace in enlaces_procesados_sesion:
            continue
        enlaces_procesados_sesion.add(enlace)

        clave_duplicado = (titulo_str.strip().lower(), organo_base)
        texto_completo = (
            f"passage: Título: {titulo_str}. Órgano: {organo}. "
            f"Tipo de contrato: {tipo_contrato}. CPV: {cpv_codigo}. "
            f"Lugar: {lugar_ejecucion}. Importe: {importe} EUR."
        )

        # ====================================================
        # VALIDACIÓN DE REGISTRO EXISTENTE
        # ====================================================
        if enlace in registros_db:
            reg_antiguo = registros_db[enlace]
            fuente_actual = str(reg_antiguo.get("fuente", ""))
            tipo_actual = reg_antiguo.get("tipo_contrato", "")
            actualizar_datos = {}

            if "Comunidad de Madrid" not in fuente_actual:
                nueva_fuente = f"{fuente_actual}, Comunidad de Madrid" if fuente_actual else "Comunidad de Madrid"
                actualizar_datos["fuente"] = nueva_fuente

            if (not tipo_actual or tipo_actual == "No especificado") and tipo_contrato != "No especificado":
                actualizar_datos["tipo_contrato"] = tipo_contrato

            es_actualizado = reg_antiguo.get("titulo") != titulo_str.strip() or reg_antiguo.get("importe") != importe
            if es_actualizado:
                actualizar_datos["es_actualizada"] = True

            if actualizar_datos:
                try:
                    supabase.table("licitaciones").update(actualizar_datos).eq("enlace", enlace).execute()
                    reg_antiguo.update(actualizar_datos)
                except Exception as e:
                    print(f"Error actualizando registro existente Madrid {enlace}: {e}")
            continue

        # ====================================================
        # REGISTRO NUEVO
        # ====================================================
        embedding = encoder.encode(texto_completo).tolist()
        es_nuevo = clave_duplicado not in registros_existentes

        elemento = {
            "titulo": titulo_str.strip(),
            "organo": organo,
            "fecha": fecha_pub,
            "importe": importe,
            "enlace": enlace,
            "texto_completo": texto_completo,
            "embedding": embedding,
            "fecha_fin": fecha_fin_str,
            "lugar_ejecucion": lugar_ejecucion,
            "cpv": cpv_codigo,
            "tipo_contrato": tipo_contrato,
            "es_novedad": es_nuevo,
            "es_actualizada": False,
            "fuente": "Comunidad de Madrid"
        }
        licitaciones_validas.append(elemento)

    # ========================================================
    # 3. LIMPIEZA AUTOMÁTICA DE CADUCADAS
    # ========================================================
    try:
        todos_db = (
            supabase.table("licitaciones")
            .select("id, enlace, fecha_fin, fuente")
            .ilike("fuente", "%Comunidad de Madrid%")
            .execute()
        )
        ids_a_borrar = []
        for item in todos_db.data:
            f_fin = item.get("fecha_fin")
            if f_fin and f_fin != "No especificada":
                try:
                    f_cierre = datetime.strptime(f_fin, "%Y-%m-%d").date()
                    if f_cierre < hoy:
                        ids_a_borrar.append(item["id"])
                except ValueError:
                    pass

        if ids_a_borrar:
            for i in range(0, len(ids_a_borrar), 50):
                lote_ids = ids_a_borrar[i:i + 50]
                supabase.table("licitaciones").delete().in_("id", lote_ids).execute()
            print(f"Eliminadas {len(ids_a_borrar)} licitaciones caducadas de Supabase.")
    except Exception as e:
        print(f"Error en la limpieza de caducadas: {e}")

    # ========================================================
    # 4. ESTADÍSTICAS
    # ========================================================
    print(f"\n--- ESTADÍSTICAS COMUNIDAD DE MADRID ---")
    print(f"Descartados por fecha caducada: {filtrados_caducados}")
    print(f"Licitaciones válidas listas para insertar: {len(licitaciones_validas)}\n")

    # ========================================================
    # 5. INSERCIÓN OPTIMIZADA
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
                    print(f"⚠️ Intento {intento}/{max_intentos} fallido para lote Madrid {num_lote}: {e}")
                    if intento < max_intentos:
                        time.sleep(2 * intento)
                    else:
                        print(f"❌ Error definitivo al subir lote Madrid {num_lote}.")

            if not exito:
                pass

        print(f"Sincronización completada con éxito. Se han subido/actualizado {subidas_exitosas} de {total_a_subir} licitaciones.")
    else:
        print("No hay licitaciones para procesar en este rango.")

if __name__ == "__main__":
    sincronizar_licitaciones_madrid()
