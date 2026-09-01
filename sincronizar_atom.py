from datetime import date, datetime
import os
import lxml.etree as ET
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client
import time
from urllib3.util.retry import Retry
import requests

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small)...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

FEEDS_ATOM = [
    {
        "nombre": "Licitaciones Generales PLACSP",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3.atom",
    },
    {
        "nombre": "Licitaciones Agregadas PLACSP",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_644/licitacionesAgregadas.atom",
    },
]

MAX_PAGINAS = 25

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cac": "urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2",
    "cac-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2",
    "cbc-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2",
}

MAPEO_NUTS = {
    "ES11": "Galicia", "ES12": "Principado de Asturias", "ES13": "Cantabria",
    "ES21": "País Vasco", "ES22": "Comunidad Foral de Navarra", "ES23": "La Rioja",
    "ES24": "Aragón", "ES30": "Comunidad de Madrid", "ES41": "Castilla y León",
    "ES42": "Castilla-La Mancha", "ES43": "Extremadura", "ES51": "Cataluña",
    "ES52": "Comunidad Valenciana", "ES53": "Illes Balears", "ES61": "Andalucía",
    "ES62": "Región de Murcia", "ES63": "Ciudad Autónoma de Ceuta",
    "ES64": "Ciudad Autónoma de Melilla", "ES70": "Canarias", "ES111": "A Coruña",
    "ES112": "Lugo", "ES113": "Ourense", "ES114": "Pontevedra", "ES120": "Asturias",
    "ES130": "Cantabria", "ES211": "Álava", "ES212": "Guipúzcoa", "ES213": "Vizcaya",
    "ES220": "Navarra", "ES230": "La Rioja", "ES241": "Huesca", "ES242": "Teruel",
    "ES243": "Zaragoza", "ES300": "Madrid", "ES411": "Ávila", "ES412": "Burgos",
    "ES413": "León", "ES414": "Palencia", "ES415": "Salamanca", "ES416": "Segovia",
    "ES417": "Soria", "ES418": "Valladolid", "ES419": "Zamora", "ES421": "Albacete",
    "ES422": "Ciudad Real", "ES423": "Cuenca", "ES424": "Guadalajara",
    "ES425": "Toledo", "ES431": "Badajoz", "ES432": "Cáceres", "ES511": "Barcelona",
    "ES512": "Girona", "ES513": "Lleida", "ES514": "Tarragona", "ES521": "Alicante",
    "ES522": "Castellón", "ES523": "Valencia", "ES531": "Eivissa i Formentera",
    "ES532": "Mallorca", "ES533": "Menorca", "ES611": "Almería", "ES612": "Cádiz",
    "ES613": "Córdoba", "ES614": "Granada", "ES615": "Huelva", "ES616": "Jaén",
    "ES617": "Málaga", "ES618": "Sevilla", "ES620": "Murcia", "ES630": "Ceuta",
    "ES640": "Melilla", "ES703": "El Hierro", "ES704": "Fuerteventura",
    "ES705": "Gran Canaria", "ES706": "La Gomera", "ES707": "La Palma",
    "ES708": "Lanzarote", "ES709": "Tenerife"
}

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def crear_sesion_robusta():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def _texto(el, xpath, ns=NS):
    nodo = el.find(xpath, ns)
    return nodo.text.strip() if nodo is not None and nodo.text else None

def limpiar_licitaciones_caducadas():
    hoy_str = date.today().strftime("%Y-%m-%d")
    try:
        supabase.table("licitaciones").delete().lt("fecha_fin", hoy_str).neq("fecha_fin", "No especificada").execute()
        print("🧹 Licitaciones caducadas por fecha fin eliminadas correctamente.")
    except Exception as e:
        print(f"⚠️ Error limpiando caducadas: {e}")

def resetear_etiquetas_diarias():
    print("🔄 Reseteando estados de novedad y actualización de ejecuciones anteriores...")
    try:
        supabase.table("licitaciones").update({"es_novedad": False, "es_actualizada": False}).eq("es_novedad", True).execute()
        supabase.table("licitaciones").update({"es_novedad": False, "es_actualizada": False}).eq("es_actualizada", True).execute()
        print("🧹 Estados reseteados con éxito.")
    except Exception as e:
        print(f"⚠️ Aviso al resetear estados anteriores: {e}")

def obtener_ultimo_enlace(nombre_feed):
    try:
        resp = supabase.table("estado_sincronizacion").select("ultimo_enlace_procesado").eq("feed_nombre", nombre_feed).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]["ultimo_enlace_procesado"]
    except Exception:
        pass
    return None

def guardar_ultimo_enlace(nombre_feed, enlace):
    try:
        supabase.table("estado_sincronizacion").upsert({
            "feed_nombre": nombre_feed,
            "ultimo_enlace_procesado": enlace
        }, on_conflict="feed_nombre").execute()
    except Exception as e:
        print(f"⚠️ No se pudo guardar el puntero de sincronización: {e}")

# ============================================================
# 3. PROCESAMIENTO DEL FEED CON CORTE INTELIGENTE POR PÁGINA 2
# ============================================================

def procesar_feed_atom_en_linea(nombre_feed, url_inicial):
    licitaciones_por_expediente = {}
    hoy = date.today()
    url_actual = url_inicial
    paginas_procesadas = 0
    sesion = crear_sesion_robusta()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    ultimo_conocido = obtener_ultimo_enlace(nombre_feed)
    enlace_pagina_2 = None
    parar_extraccion = False

    estados_cerrados = ["EV", "ADJ", "RES", "ANUL", "FOR", "AS", "RE", "CAN"]

    while url_actual and paginas_procesadas < MAX_PAGINAS and not parar_extraccion:
        paginas_procesadas += 1

        # Si llegamos a la página 2 y coincide con la URL guardada en la ejecución anterior, detenemos
        if paginas_procesadas == 2 and ultimo_conocido and url_actual == ultimo_conocido:
            print(f"🛑 Encontrada la página 2 ya conocida ({url_actual}). Deteniendo lectura de páginas históricas.")
            break

        print(f"📄 PÁGINA {paginas_procesadas}/{MAX_PAGINAS} - {url_actual}")
        try:
            resp = sesion.get(url_actual, headers=headers, timeout=30)
            if resp.status_code != 200:
                break

            parser = ET.XMLParser(recover=True)
            root = ET.fromstring(resp.content, parser=parser)
            entries = root.findall("atom:entry", NS) or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if not entries:
                break

            # Guardamos la URL exacta de la página 2 cuando estemos en ella
            if paginas_procesadas == 2:
                enlace_pagina_2 = url_actual

            for index, entry in enumerate(entries):
                enlace_el = entry.find("atom:link", NS)
                enlace = enlace_el.get("href") if enlace_el is not None else ""
                if enlace and "contrataciondelestado.es" in enlace:
                    enlace = enlace.replace("contrataciondelestado.es", "contrataciondelsectorpublico.gob.es")
                enlace = enlace.strip()

                codigo_estado = "PUB"
                try:
                    estado_el = entry.find(".//cbc-place-ext:ContractFolderStatusCode", NS) or entry.find(".//cbc:ContractFolderStatusCode", NS)
                    if estado_el is not None and estado_el.text:
                        codigo_estado = estado_el.text.strip().upper()
                except Exception:
                    pass

                if codigo_estado in estados_cerrados:
                    try:
                        supabase.table("licitaciones").delete().eq("enlace", enlace).execute()
                    except Exception:
                        pass
                    continue

                txt_updated = _texto(entry, "atom:updated")
                txt_published = _texto(entry, "atom:published")
                txt_fecha = txt_updated or txt_published
                if not txt_fecha:
                    continue

                fecha_str = txt_fecha.split("T")[0]

                end_date_el = entry.find(".//cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod/cbc:EndDate", NS)
                fecha_fin_str = "No especificada"
                if end_date_el is not None and end_date_el.text:
                    fecha_fin_str = end_date_el.text.strip()[:10]
                    try:
                        fecha_fin = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                        if fecha_fin < hoy:
                            print(f"🗑️ Licitación caducada detectada: {enlace}")
                            try:
                                supabase.table("licitaciones").delete().eq("enlace", enlace).execute()
                            except Exception as e:
                                print(f"⚠️ No se pudo eliminar la licitación caducada: {e}")
                            continue
                    except Exception:
                        pass

                titulo = _texto(entry, "atom:title") or "Sin título"
                expediente_id = _texto(entry, ".//cbc:ContractFolderID", NS) or enlace

                cpv_codigo = "No especificado"
                try:
                    cpv_elements = entry.findall(".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cac:RequiredCommodityClassification/cbc:ItemClassificationCode", NS) or entry.findall(".//cbc:ItemClassificationCode", NS)
                    if cpv_elements:
                        cpv_codigo = ", ".join([el.text.strip() for el in cpv_elements if el.text])
                except Exception:
                    pass

                lugar_ejecucion = "No especificado"
                try:
                    lugar_el = entry.find(".//cac:ProcurementProject/cac:RealizedLocation/cbc:CountrySubentity", NS)
                    if lugar_el is not None and lugar_el.text:
                        lugar_ejecucion = lugar_el.text.strip()
                    else:
                        lugar_el = entry.find(".//cac:ProcurementProject/cac:RealizedLocation/cbc:CountrySubentityCode", NS)
                        if lugar_el is not None and lugar_el.text:
                            lugar_ejecucion = MAPEO_NUTS.get(lugar_el.text.strip(), lugar_el.text.strip())
                except Exception:
                    pass

                importe = 0.0
                try:
                    presupuesto_el = entry.find(".//cac:BudgetAmount/cbc:EstimatedOverallContractAmount", NS) or entry.find(".//cac:BudgetAmount/cbc:TaxExclusiveAmount", NS) or entry.find(".//cac:BudgetAmount/cbc:TotalAmount", NS)
                    if presupuesto_el is not None and presupuesto_el.text:
                        importe = float(presupuesto_el.text.strip().replace(",", "."))
                except Exception:
                    pass

                organo = "Órgano desconocido"
                rutas_organo = [
                    ".//cac-place-ext:LocatedContractingParty//cac:PartyName//cbc:Name",
                    ".//cac:ContractingParty//cac:PartyName//cbc:Name",
                    ".//cac:TenderingParty//cac:PartyName//cbc:Name",
                    ".//cac:ContractingParty//cac:Party//cac:PartyName//cbc:Name",
                    ".//cbc:PartyName//cbc:Name"
                ]
                for ruta in rutas_organo:
                    organo_el = entry.find(ruta, NS)
                    if organo_el is not None and organo_el.text and organo_el.text.strip():
                        organo = organo_el.text.strip()
                        break

                descripcion = _texto(entry, ".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cbc:Name", NS) or _texto(entry, ".//cac:ProcurementProject/cbc:Description", NS) or ""
                texto_evaluacion = f"passage: Título: {titulo}. Objeto: {descripcion}. Órgano: {organo}. CPV: {cpv_codigo}. Lugar: {lugar_ejecucion}. Importe: {importe} EUR."

                licitacion_data = {
                    "enlace": enlace,
                    "titulo": titulo.strip(),
                    "organo": organo.strip(),
                    "fecha": fecha_str,
                    "importe": importe,
                    "cpv": cpv_codigo,
                    "lugar_ejecucion": lugar_ejecucion,
                    "fecha_fin": fecha_fin_str,
                    "texto_completo": texto_evaluacion,
                    "fuente": nombre_feed,
                    "_atom_updated": txt_updated or txt_fecha,
                }

                if expediente_id not in licitaciones_por_expediente or licitacion_data["_atom_updated"] > licitaciones_por_expediente[expediente_id]["_atom_updated"]:
                    licitaciones_por_expediente[expediente_id] = licitacion_data

            next_link_el = root.find("atom:link[@rel='next']", NS) or root.find(".//{http://www.w3.org/2005/Atom}link[@rel='next']")
            url_actual = next_link_el.get("href") if next_link_el is not None else None
            if url_actual and "contrataciondelestado.es" in url_actual:
                url_actual = url_actual.replace("contrataciondelestado.es", "contrataciondelsectorpublico.gob.es")

            time.sleep(1)
        except Exception as e:
            print(f"❌ Error de red: {e}. Reintentando...")
            time.sleep(10)
            continue

    if enlace_pagina_2:
        guardar_ultimo_enlace(nombre_feed, enlace_pagina_2)

    for item in licitaciones_por_expediente.values():
        item.pop("_atom_updated", None)

    return list(licitaciones_por_expediente.values())

# ============================================================
# 4. ORQUESTACIÓN Y PROCESAMIENTO
# ============================================================

limpiar_licitaciones_caducadas()
resetear_etiquetas_diarias()

todas_licitaciones = []
for feed in FEEDS_ATOM:
    print(f"🌐 PROCESANDO FEED: {feed['nombre']}")
    lics = procesar_feed_atom_en_linea(feed["nombre"], feed["url"])
    todas_licitaciones.extend(lics)

print("🔍 Consultando base de datos existente para marcar novedades y actualizaciones...")
existentes_resp = supabase.table("licitaciones").select("enlace, fecha").execute()
mapa_existentes = {item["enlace"]: item["fecha"] for item in existentes_resp.data}

for lic in todas_licitaciones:
    enlace = lic["enlace"]
    if enlace not in mapa_existentes:
        lic["es_novedad"] = True
        lic["es_actualizada"] = False
    else:
        lic["es_novedad"] = False
        if lic["fecha"] > mapa_existentes[enlace]:
            lic["es_actualizada"] = True
        else:
            lic["es_actualizada"] = False

# ============================================================
# 5. SUBIDA A SUPABASE
# ============================================================

if todas_licitaciones:
    print("🚀 Generando embeddings y subiendo a Supabase...")
    for i, lic in enumerate(todas_licitaciones):
        lic["embedding"] = encoder.encode(lic["texto_completo"]).tolist()
        try:
            supabase.table("licitaciones").upsert(lic, on_conflict="enlace").execute()
            if (i + 1) % 50 == 0:
                print(f"    -> Subidas {i + 1} de {len(todas_licitaciones)}...")
        except Exception as e:
            print(f"❌ Error subiendo a Supabase: {e}")
    print("✅ ¡Carga, control de paginación y reseteo completados con éxito!")
else:
    print("ℹ️ No hay licitaciones nuevas que procesar en esta ejecución.")
