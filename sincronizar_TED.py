from datetime import datetime, date, timedelta
import os
import time
import requests
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small) para TED...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

# ============================================================
# MAPEO NUTS TED
# ============================================================

MAPEO_NUTS_TED = {
    "ES111": "A Coruña",
    "ES112": "Lugo",
    "ES113": "Ourense",
    "ES114": "Pontevedra",
    "ES120": "Asturias",
    "ES130": "Cantabria",
    "ES211": "Álava/Araba",
    "ES212": "Gipuzkoa",
    "ES213": "Bizkaia",
    "ES220": "La Rioja",
    "ES230": "Navarra",
    "ES241": "Huesca",
    "ES242": "Teruel",
    "ES243": "Zaragoza",
    "ES300": "Madrid",
    "ES411": "Ávila",
    "ES412": "Burgos",
    "ES413": "León",
    "ES414": "Palencia",
    "ES415": "Salamanca",
    "ES416": "Segovia",
    "ES417": "Soria",
    "ES418": "Valladolid",
    "ES419": "Zamora",
    "ES421": "Albacete",
    "ES422": "Ciudad Real",
    "ES423": "Cuenca",
    "ES424": "Guadalajara",
    "ES425": "Toledo",
    "ES431": "Badajoz",
    "ES432": "Cáceres",
    "ES511": "Barcelona",
    "ES512": "Girona",
    "ES513": "Lleida",
    "ES514": "Tarragona",
    "ES521": "Alicante/Alacant",
    "ES522": "Castellón/Castelló",
    "ES523": "Valencia/València",
    "ES531": "Eivissa y Formentera",
    "ES532": "Mallorca",
    "ES533": "Menorca",
    "ES611": "Almería",
    "ES612": "Cádiz",
    "ES613": "Córdoba",
    "ES614": "Granada",
    "ES615": "Huelva",
    "ES616": "Jaén",
    "ES617": "Málaga",
    "ES618": "Sevilla",
    "ES620": "Murcia",
    "ES630": "Ceuta",
    "ES640": "Melilla",
    "ES703": "El Hierro",
    "ES704": "Fuerteventura",
    "ES705": "Gran Canaria",
    "ES706": "La Gomera",
    "ES707": "La Palma",
    "ES708": "Lanzarote",
    "ES709": "Tenerife",
    "ES1": "Noroeste (España)",
    "ES2": "Noreste (España)",
    "ES3": "Comunidad de Madrid (España)",
    "ES4": "Centro (España)",
    "ES5": "Este (España)",
    "ES6": "Sur (España)",
    "ES7": "Canarias (España)"
}

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def mapear_lugar(lugar_str):
    if not lugar_str or lugar_str == "No especificado":
        return lugar_str
    elementos = [e.strip() for e in lugar_str.split(",")]
    elementos_mapeados = [MAPEO_NUTS_TED.get(el, el) for el in elementos]
    return ", ".join(dict.fromkeys(elementos_mapeados))

def limpiar_titulo(titulo):
    if isinstance(titulo, list):
        titulo = " ".join([str(x) for x in titulo if x])
    elif not isinstance(titulo, str):
        titulo = str(titulo) if titulo else ""
    titulo = titulo.strip()
    for separador in [" – ", " - ", " — "]:
        if separador in titulo:
            partes = titulo.split(separador)
            if len(partes) > 1:
                titulo = partes[-1].strip()
                break
    return titulo.strip()

def limpiar_organo(organo):
    if isinstance(organo, list):
        organo = organo[0] if organo else ""
    elif isinstance(organo, dict):
        organo = organo.get("eng") or next(iter(organo.values()), "")
    texto = str(organo).strip()
    for char in ["[", "]", "'", '"']:
        texto = texto.replace(char, "")
    return texto.strip()

def limpiar_tipo_contrato(tipo_raw):
    """
    Mapea el contract-nature de TED a formatos estandarizados.
    """
    if isinstance(tipo_raw, list):
        tipo_raw = tipo_raw[0] if tipo_raw else ""
    elif isinstance(tipo_raw, dict):
        tipo_raw = tipo_raw.get("eng") or next(iter(tipo_raw.values()), "")
    t_str = str(tipo_raw).strip().lower()
    if "supplies" in t_str or "suministro" in t_str:
        return "Suministros"
    elif "services" in t_str or "servicio" in t_str:
        return "Servicios"
    elif "works" in t_str or "obra" in t_str:
        return "Obras"
    elif t_str and t_str != "no especificado":
        return str(tipo_raw).strip().capitalize()
    return "No especificado"

def procesar_campo(campo, es_lista=False):
    if isinstance(campo, list):
        limpios = [str(x) for x in campo if x]
        if es_lista:
            return list(dict.fromkeys(limpios))
        return limpios[0] if limpios else "No especificado"
    elif isinstance(campo, dict):
        return campo.get("eng") or next(iter(campo.values()), "No especificado")
    return str(campo) if campo else "No especificado"

# ============================================================
# 3. CONSULTA A LA API DE TED
# ============================================================

def consultar_ted_api_scroll():
    url = "https://api.ted.europa.eu/v3/notices/search"
    fields_solicitados = [
        "publication-number",
        "contract-title",
        "notice-title",
        "organisation-name-buyer",
        "publication-date",
        "deadline-receipt-request",
        "place-of-performance",
        "classification-cpv",
        "description-proc",
        "total-value",
        "total-value-cur",
        "notice-type",
        "form-type",
        "contract-nature"
    ]
    hoy = date.today()
    fecha_inicio = (hoy - timedelta(days=7)).strftime("%Y%m%d")
    fecha_fin_str = hoy.strftime("%Y%m%d")
    todas_licitaciones = []
    iteration_next_token = None
    limit = 250

    print(f"Consultando la API de TED para España ({fecha_inicio} a {fecha_fin_str})...")

    while True:
        payload = {
            "query": f"publication-date >= '{fecha_inicio}' AND publication-date <= '{fecha_fin_str}' AND buyer-country = 'ESP'",
            "fields": fields_solicitados,
            "paginationMode": "ITERATION",
            "limit": limit
        }
        if iteration_next_token:
            payload["iterationNextToken"] = iteration_next_token
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                notices = data.get("notices", [])
                if not notices:
                    break
                todas_licitaciones.extend(notices)
                print(f"Lote TED descargado ({len(notices)} avisos). Acumulado: {len(todas_licitaciones)}")
                iteration_next_token = data.get("iterationNextToken")
                if not iteration_next_token or len(notices) < limit:
                    break
                time.sleep(0.3)
            elif response.status_code == 429:
                print("Límite de peticiones TED alcanzado (429). Esperando 5 segundos...")
                time.sleep(5)
                continue
            else:
                print(f"Error API TED HTTP {response.status_code}: {response.text}")
                break
        except Exception as e:
            print(f"Excepción conectando con TED: {e}")
            break
    return todas_licitaciones

# ============================================================
# 4. SINCRONIZACIÓN TED
# ============================================================

def sincronizar_licitaciones_ted():
    hoy = datetime.now().date()

    # ========================================================
    # 1. OBTENER AVISOS DE LA API
    # ========================================================
    lics_ted = consultar_ted_api_scroll()
    if not lics_ted:
        print("ℹ️ No se obtuvieron avisos de la API de TED.")
        return
    print(f"\nTotal de avisos obtenidos de TED: {len(lics_ted)}")

    # ========================================================
    # 2. CARGAR REGISTROS EXISTENTES
    # ========================================================
    print("\nCargando registros existentes desde Supabase...")
    try:
        existentes_resp = supabase.table("licitaciones").select(
            "id, enlace, titulo, organo, fecha, importe, fuente, fecha_fin, es_novedad, es_actualizada"
        ).execute()

        mapa_enlaces = {
            item["enlace"]: item
            for item in existentes_resp.data
            if item.get("enlace")
        }
        registros_existentes = {
            (
                limpiar_titulo(item.get("titulo", "")).lower(),
                limpiar_organo(item.get("organo", "")).lower()
            )
            for item in existentes_resp.data
        }

        # IDs DE REGISTROS TED CON FLAGS ACTIVAS
        ids_flags_ted = []
        for item in existentes_resp.data:
            fuente_item = str(item.get("fuente", ""))
            if "ted" in fuente_item.casefold() and (item.get("es_novedad") is True or item.get("es_actualizada") is True):
                ids_flags_ted.append(item["id"])

        print(f"Registros totales cargados desde Supabase: {len(existentes_resp.data)}")
    except Exception as e:
        print(f"❌ Error conectando con Supabase: {e}")
        return

    # ========================================================
    # 3. RESETEAR ETIQUETAS ANTERIORES DE TED
    # ========================================================
    if ids_flags_ted:
        print(f"\nReseteando etiquetas anteriores de {len(ids_flags_ted)} registros de TED...")
        tamano_reset = 25
        max_intentos_reset = 3
        reset_correcto = True
        total_reseteadas = 0

        for i in range(0, len(ids_flags_ted), tamano_reset):
            lote_ids = ids_flags_ted[i:i + tamano_reset]
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
            print(f"Etiquetas anteriores de TED reseteadas correctamente: {total_reseteadas} registros.")
        else:
            print("⚠️ Aviso: no se pudieron resetear todas las etiquetas anteriores de TED.")
    else:
        print("\nNo hay etiquetas anteriores de TED que resetear.")

    # ========================================================
    # 4. LIMPIEZA DE REGISTROS TED CADUCADOS
    # ========================================================
    try:
        todos_db = supabase.table("licitaciones").select(
            "id, enlace, fecha_fin, fuente"
        ).ilike("fuente", "%TED%").execute()

        ids_a_borrar = []
        for item in todos_db.data:
            f_fin = item.get("fecha_fin")
            if not f_fin or f_fin == "No especificada":
                continue
            try:
                f_cierre = datetime.strptime(f_fin, "%Y-%m-%d").date()
                if f_cierre < hoy:
                    fuente_actual = str(item.get("fuente", ""))
                    # Si TED es la única fuente, eliminar
                    fuentes = [f.strip() for f in fuente_actual.split(",") if f.strip()]
                    fuentes_restantes = [f for f in fuentes if f.casefold() != "ted"]

                    if not fuentes_restantes:
                        ids_a_borrar.append(item["id"])
                    else:
                        # Si hay otras fuentes, quitar TED
                        nueva_fuente = ", ".join(fuentes_restantes)
                        try:
                            supabase.table("licitaciones").update({
                                "fuente": nueva_fuente
                            }).eq("id", item["id"]).execute()
                        except Exception as e:
                            print(f"Error quitando TED de fuente combinada {item['id']}: {e}")
            except ValueError:
                pass

        # BORRAR EN LOTES
        if ids_a_borrar:
            for i in range(0, len(ids_a_borrar), 50):
                lote_ids = ids_a_borrar[i:i + 50]
                supabase.table("licitaciones").delete().in_("id", lote_ids).execute()
            print(f"🧹 Eliminadas {len(ids_a_borrar)} licitaciones TED caducadas.")
        else:
            print("🧹 No hay licitaciones TED caducadas para eliminar.")
    except Exception as e:
        print(f"⚠️ Aviso al limpiar caducadas de TED: {e}")

    # ========================================================
    # 5. PROCESAR AVISOS
    # ========================================================
    licitaciones_validas = []
    filtrados_adjudicados = 0
    filtrados_caducados = 0
    filtrados_veat = 0
    filtrados_duplicados_placsp = 0
    total_actualizadas = 0
    total_nuevas = 0
    enlaces_procesados_sesion = set()

    for aviso in lics_ted:
        # ENLACE
        num = aviso.get("publication-number")
        if not num:
            continue
        enlace = f"https://ted.europa.eu/en/notice/-/detail/{num}"
        if enlace in enlaces_procesados_sesion:
            continue
        enlaces_procesados_sesion.add(enlace)

        # TÍTULO
        c_title = procesar_campo(aviso.get("contract-title"))
        n_title = procesar_campo(aviso.get("notice-title"))
        titulo_bruto = c_title if c_title != "No especificado" else (n_title if n_title != "No especificado" else "Sin título")
        titulo_limpio = limpiar_titulo(titulo_bruto)

        # ÓRGANO
        organo_bruto = aviso.get("organisation-name-buyer")
        organo_limpio = limpiar_organo(organo_bruto)

        # TIPO DE AVISO
        n_type = str(aviso.get("notice-type", "")).lower()
        f_type = str(aviso.get("form-type", "")).lower()

        # FILTRAR ADJUDICADOS / RESULTADOS
        es_adjudicado = n_type.startswith("can-") or "award" in n_type or f_type == "result"
        if es_adjudicado:
            filtrados_adjudicados += 1
            try:
                if enlace in mapa_enlaces:
                    supabase.table("licitaciones").delete().eq("enlace", enlace).execute()
            except Exception:
                pass
            continue

        # FILTRAR VEAT
        es_veat = n_type.startswith("dir-awa-pre") or "dir-awa-pre" in n_type or "veat" in n_type
        if es_veat:
            filtrados_veat += 1
            continue

        # FECHA DE CIERRE
        fechas_cierre = procesar_campo(aviso.get("deadline-receipt-request"), es_lista=True)
        fecha_fin_str = "No especificada"
        if fechas_cierre != "No especificado":
            limite_str = fechas_cierre[0] if isinstance(fechas_cierre, list) else str(fechas_cierre)
            fecha_fin_str = limite_str[:10]
            try:
                fecha_cierre_date = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                if fecha_cierre_date < hoy:
                    filtrados_caducados += 1
                    continue
            except ValueError:
                pass

        # FECHA DE PUBLICACIÓN
        fecha_pub_raw = aviso.get("publication-date", "")
        fecha_str = fecha_pub_raw[:10] if fecha_pub_raw else date.today().strftime("%Y-%m-%d")

        # IMPORTE
        importe_raw = aviso.get("total-value")
        if isinstance(importe_raw, dict):
            importe_val = importe_raw.get("value") or next(iter(importe_raw.values()), None)
        elif isinstance(importe_raw, list):
            importe_val = importe_raw[0] if importe_raw else None
        else:
            importe_val = importe_raw
        try:
            importe = float(importe_val) if importe_val is not None else 0.0
        except (ValueError, TypeError):
            importe = 0.0

        # CPV
        cpvs = ", ".join(procesar_campo(aviso.get("classification-cpv"), es_lista=True))

        # LUGAR
        lugares_raw = ", ".join(procesar_campo(aviso.get("place-of-performance"), es_lista=True))
        lugares = mapear_lugar(lugares_raw)

        # DESCRIPCIÓN
        descripcion = procesar_campo(aviso.get("description-proc"))

        # TIPO DE CONTRATO
        tipo_contrato = limpiar_tipo_contrato(aviso.get("contract-nature"))

        # TEXTO COMPLETO
        texto_completo = f"passage: Título: {titulo_limpio}. Objeto: {descripcion}. Órgano: {organo_limpio}. Tipo de contrato: {tipo_contrato}. CPV: {cpvs}. Lugar: {lugares}. Importe: {importe} EUR."
        clave_duplicado = (titulo_limpio.lower(), organo_limpio.lower())

        # ====================================================
        # 6. COMPROBAR SI EXISTE
        # ====================================================
        es_novedad = False
        es_actualizada = False
        fuente_final = "TED"

        if enlace in mapa_enlaces:
            reg_existente = mapa_enlaces[enlace]
            fuente_existente = str(reg_existente.get("fuente", "TED"))
            if "ted" not in fuente_existente.casefold():
                fuente_final = f"{fuente_existente}, TED" if fuente_existente else "TED"
            else:
                fuente_final = fuente_existente

            es_actualizado = (
                reg_existente.get("titulo") != titulo_limpio
                or reg_existente.get("importe") != importe
                or reg_existente.get("fecha_fin") != fecha_fin_str
            )
            if es_actualizado:
                es_actualizada = True
                total_actualizadas += 1
        else:
            if clave_duplicado in registros_existentes:
                filtrados_duplicados_placsp += 1
                continue
            else:
                es_novedad = True
                total_nuevas += 1

        # CREAR ELEMENTO
        embedding = encoder.encode(texto_completo).tolist()
        elemento = {
            "titulo": titulo_limpio,
            "organo": organo_limpio,
            "fecha": fecha_str,
            "importe": importe,
            "enlace": enlace,
            "texto_completo": texto_completo,
            "embedding": embedding,
            "fecha_fin": fecha_fin_str,
            "lugar_ejecucion": lugares,
            "cpv": cpvs,
            "tipo_contrato": tipo_contrato,
            "es_novedad": es_novedad,
            "es_actualizada": es_actualizada,
            "fuente": fuente_final
        }
        licitaciones_validas.append(elemento)

    # ========================================================
    # 7. ESTADÍSTICAS
    # ========================================================
    print("\n============================================================")
    print("ESTADÍSTICAS TED")
    print("============================================================")
    print(f"Duplicados evitados (ya estaban en PLACSP): {filtrados_duplicados_placsp}")
    print(f"Adjudicados/resultados filtrados: {filtrados_adjudicados}")
    print(f"VEAT filtrados: {filtrados_veat}")
    print(f"Caducados filtrados: {filtrados_caducados}")
    print(f"Nuevas: {total_nuevas}")
    print(f"Actualizadas: {total_actualizadas}")
    print("============================================================")

    # ========================================================
    # 8. INSERCIÓN EN SUPABASE
    # ========================================================
    if not licitaciones_validas:
        print("ℹ️ No hay licitaciones de TED válidas para insertar.")
        return

    print(f"📤 Insertando {len(licitaciones_validas)} licitaciones de TED en Supabase...")
    tamano_lote = 5
    subidas_exitosas = 0
    max_intentos = 3

    for i in range(0, len(licitaciones_validas), tamano_lote):
        lote = licitaciones_validas[i:i + tamano_lote]
        num_lote = (i // tamano_lote) + 1
        exito = False

        for intento in range(1, max_intentos + 1):
            try:
                supabase.table("licitaciones").upsert(lote, on_conflict="enlace").execute()
                subidas_exitosas += len(lote)
                print(f"Progreso TED: {subidas_exitosas}/{len(licitaciones_validas)} procesadas...")
                exito = True
                break
            except Exception as e:
                print(f"⚠️ Intento {intento}/{max_intentos} fallido para lote TED {num_lote}: {e}")
                if intento < max_intentos:
                    time.sleep(2 * intento)
                else:
                    print(f"❌ Error definitivo al subir lote TED {num_lote}.")
        if not exito:
            pass

    print(f"Sincronización TED completada. Subidas/actualizadas: {subidas_exitosas} de {len(licitaciones_validas)}.")

if __name__ == "__main__":
    sincronizar_licitaciones_ted()
