from datetime import date, datetime
import os
import re
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
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_1044/PlataformasAgregadasSinMenores.atom",
    },
]

MAX_PAGINAS = 15

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
    "ES111": "A Coruña",
    "ES112": "Lugo",
    "ES113": "Ourense",
    "ES114": "Pontevedra",
    "ES120": "Asturias",
    "ES130": "Cantabria",
    "ES211": "Álava",
    "ES212": "Guipúzcoa",
    "ES213": "Vizcaya",
    "ES220": "Navarra",
    "ES230": "La Rioja",
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
    "ES521": "Alicante",
    "ES522": "Castellón",
    "ES523": "Valencia",
    "ES531": "Eivissa i Formentera",
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
    "ES709": "Tenerife"
}

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def crear_sesion_robusta():
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retries)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


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


# ============================================================
# FUNCIONES PARA MANEJAR FUENTES COMBINADAS
# ============================================================

def normalizar_fuentes(fuente):
    """
    Convierte una cadena de fuentes separadas por comas
    en una lista normalizada.

    Ejemplo:
        "TED, Licitaciones Generales PLACSP"

    devuelve:
        ["TED", "Licitaciones Generales PLACSP"]
    """

    if not fuente:
        return []

    return [
        f.strip()
        for f in str(fuente).split(",")
        if f.strip()
    ]


def contiene_fuente(fuente_actual, nombre_fuente):
    """
    Comprueba si nombre_fuente está realmente entre las
    fuentes del registro.

    Evita problemas de coincidencias por substring.
    """

    fuentes = normalizar_fuentes(fuente_actual)

    return any(
        f.casefold() == nombre_fuente.casefold()
        for f in fuentes
    )


def añadir_fuente(fuente_actual, nueva_fuente):
    """
    Añade una fuente a la lista de fuentes del registro
    solo si todavía no existe.
    """

    fuentes = normalizar_fuentes(fuente_actual)

    if any(
        f.casefold() == nueva_fuente.casefold()
        for f in fuentes
    ):
        return fuente_actual

    fuentes.append(nueva_fuente)

    return ", ".join(fuentes)


def quitar_fuente(fuente_actual, fuente_a_quitar):
    """
    Elimina una fuente concreta de una cadena de fuentes.
    """

    fuentes = normalizar_fuentes(fuente_actual)

    fuentes = [
        f for f in fuentes
        if f.casefold() != fuente_a_quitar.casefold()
    ]

    return ", ".join(fuentes)


def traducir_tipo_contrato(codigo_raw):
    """Traduce o mapea el código o texto del tipo de contrato al estándar."""

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

    return mapping_texto.get(
        limpio,
        str(codigo_raw).capitalize()
    )


def limpiar_licitaciones_caducadas():
    hoy_str = date.today().strftime("%Y-%m-%d")

    try:
        supabase.table("licitaciones") \
            .delete() \
            .lt("fecha_fin", hoy_str) \
            .neq("fecha_fin", "No especificada") \
            .execute()

        print(
            "🧹 Licitaciones caducadas por fecha fin "
            "eliminadas correctamente."
        )

    except Exception as e:
        print(f"⚠️ Error limpiando caducadas: {e}")


# ============================================================
# RESETEO DE FLAGS
# ============================================================

def resetear_etiquetas_por_fuente(nombre_feed):
    """
    Resetea es_novedad y es_actualizada para todos los registros
    cuya fuente contiene nombre_feed, incluso cuando la fuente
    está combinada con otras.

    Ejemplo válido:

        TED, Licitaciones Generales PLACSP

    Si nombre_feed es:

        Licitaciones Generales PLACSP

    el registro será detectado correctamente.
    """

    print(
        f"🔄 Reseteando estados para la fuente: {nombre_feed}..."
    )

    try:
        # --------------------------------------------------------
        # Cargar registros que tengan información de fuente
        # y alguna de las dos flags activadas
        # --------------------------------------------------------

        res = supabase.table("licitaciones").select(
            "id,fuente,es_novedad,es_actualizada"
        ).execute()

        registros = res.data or []

        ids_a_resetear = []

        for item in registros:

            fuente = str(
                item.get("fuente", "")
            ).strip()

            # Solo nos interesan registros que realmente
            # pertenezcan a este feed
            if not contiene_fuente(fuente, nombre_feed):
                continue

            # Solo hace falta actualizar aquellos que tengan
            # alguna de las dos etiquetas activada
            if (
                item.get("es_novedad") is True
                or item.get("es_actualizada") is True
            ):
                ids_a_resetear.append(item["id"])

        print(
            f"   Registros con flags activadas para "
            f"'{nombre_feed}': {len(ids_a_resetear)}"
        )

        # --------------------------------------------------------
        # Resetear en lotes
        # --------------------------------------------------------

        tamano_lote = 25

        reseteados = 0

        for i in range(
            0,
            len(ids_a_resetear),
            tamano_lote
        ):

            lote_ids = ids_a_resetear[
                i:i + tamano_lote
            ]

            max_intentos = 3

            for intento in range(
                1,
                max_intentos + 1
            ):

                try:
                    supabase.table("licitaciones") \
                        .update({
                            "es_novedad": False,
                            "es_actualizada": False
                        }) \
                        .in_("id", lote_ids) \
                        .execute()

                    reseteados += len(lote_ids)

                    break

                except Exception as e:

                    print(
                        f"⚠️ Error reseteando lote "
                        f"{i // tamano_lote + 1}, "
                        f"intento {intento}/{max_intentos}: {e}"
                    )

                    if intento < max_intentos:
                        time.sleep(2 * intento)

        print(
            f"🧹 Estados reseteados con éxito para "
            f"{nombre_feed}: {reseteados}"
        )

    except Exception as e:
        print(
            f"⚠️ Aviso al resetear estados para "
            f"{nombre_feed}: {e}"
        )


# ============================================================
# 3. PROCESAMIENTO Y SINCRONIZACIÓN DE FEEDS PLACSP
# ============================================================

def procesar_y_sincronizar_feed(nombre_feed, url_inicial):

    hoy = date.today()
    url_actual = url_inicial
    paginas_procesadas = 0

    sesion = crear_sesion_robusta()

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
    }

    estados_cerrados = [
        "EV",
        "ADJ",
        "RES",
        "ANUL",
        "FOR",
        "AS",
        "RE",
        "CAN"
    ]

    licitaciones_por_expediente = {}

    print(
        f"\n🌐 PROCESANDO FEED: {nombre_feed}"
    )

    # ========================================================
    # RESETEAR FLAGS ANTES DE PROCESAR EL FEED
    # ========================================================

    resetear_etiquetas_por_fuente(nombre_feed)

    # ========================================================
    # DESCARGA DE FEEDS
    # ========================================================

    while (
        url_actual
        and paginas_procesadas < MAX_PAGINAS
    ):

        paginas_procesadas += 1

        print(
            f"  📄 PÁGINA "
            f"{paginas_procesadas}/{MAX_PAGINAS} "
            f"- {url_actual}"
        )

        try:

            resp = sesion.get(
                url_actual,
                headers=headers,
                timeout=30
            )

            if resp.status_code != 200:

                print(
                    f"  ⚠️ Error HTTP "
                    f"{resp.status_code} "
                    f"al descargar la página."
                )

                break

            parser = ET.XMLParser(
                recover=True
            )

            root = ET.fromstring(
                resp.content,
                parser=parser
            )

            entries = (
                root.findall(
                    "atom:entry",
                    NS
                )
                or root.findall(
                    ".//{http://www.w3.org/2005/Atom}entry"
                )
            )

            if not entries:

                print(
                    "  ℹ️ No se encontraron más "
                    "entradas en esta página."
                )

                break

            # =================================================
            # PROCESAR ENTRIES
            # =================================================

            for entry in entries:

                # -------------------------------------------------
                # ENLACE
                # -------------------------------------------------

                enlace_el = entry.find(
                    "atom:link",
                    NS
                )

                enlace = (
                    enlace_el.get("href")
                    if enlace_el is not None
                    else ""
                )

                if (
                    enlace
                    and "contrataciondelestado.es"
                    in enlace
                ):
                    enlace = enlace.replace(
                        "contrataciondelestado.es",
                        "contrataciondelsectorpublico.gob.es"
                    )

                enlace = enlace.strip()

                if not enlace:
                    continue

                # -------------------------------------------------
                # ESTADO
                # -------------------------------------------------

                codigo_estado = "PUB"

                try:

                    estado_el = entry.find(
                        ".//cbc-place-ext:"
                        "ContractFolderStatusCode",
                        NS
                    )

                    if estado_el is None:

                        estado_el = entry.find(
                            ".//cbc:ContractFolderStatusCode",
                            NS
                        )

                    if (
                        estado_el is not None
                        and estado_el.text
                    ):
                        codigo_estado = (
                            estado_el.text
                            .strip()
                            .upper()
                        )

                except Exception:
                    pass

                # -------------------------------------------------
                # ELIMINAR LICITACIONES CERRADAS
                # -------------------------------------------------

                if codigo_estado in estados_cerrados:

                    try:

                        supabase.table(
                            "licitaciones"
                        ).delete() \
                         .eq(
                             "enlace",
                             enlace
                         ).execute()

                    except Exception:
                        pass

                    continue

                # -------------------------------------------------
                # FECHA
                # -------------------------------------------------

                txt_updated = _texto(
                    entry,
                    "atom:updated"
                )

                txt_published = _texto(
                    entry,
                    "atom:published"
                )

                txt_fecha = (
                    txt_updated
                    or txt_published
                )

                if not txt_fecha:
                    continue

                fecha_str = txt_fecha.split("T")[0]

                # -------------------------------------------------
                # FECHA FIN
                # -------------------------------------------------

                end_date_el = entry.find(
                    ".//cac:TenderingProcess/"
                    "cac:TenderSubmissionDeadlinePeriod/"
                    "cbc:EndDate",
                    NS
                )

                fecha_fin_str = "No especificada"

                if (
                    end_date_el is not None
                    and end_date_el.text
                ):

                    fecha_fin_str = (
                        end_date_el.text
                        .strip()[:10]
                    )

                    try:

                        fecha_fin = datetime.strptime(
                            fecha_fin_str,
                            "%Y-%m-%d"
                        ).date()

                        if fecha_fin < hoy:

                            try:

                                supabase.table(
                                    "licitaciones"
                                ).delete() \
                                 .eq(
                                     "enlace",
                                     enlace
                                 ).execute()

                            except Exception:
                                pass

                            continue

                    except Exception:
                        pass

                # -------------------------------------------------
                # TÍTULO
                # -------------------------------------------------

                titulo = (
                    _texto(
                        entry,
                        "atom:title"
                    )
                    or "Sin título"
                )

                # -------------------------------------------------
                # EXPEDIENTE
                # -------------------------------------------------

                expediente_id = (
                    _texto(
                        entry,
                        ".//cbc:ContractFolderID",
                        NS
                    )
                    or enlace
                )

                # -------------------------------------------------
                # CPV
                # -------------------------------------------------

                cpv_codigo = "No especificado"

                try:

                    cpv_elements = (
                        entry.findall(
                            ".//cac-place-ext:"
                            "ContractFolderStatus/"
                            "cac:ProcurementProject/"
                            "cac:RequiredCommodityClassification/"
                            "cbc:ItemClassificationCode",
                            NS
                        )
                        or entry.findall(
                            ".//cbc:ItemClassificationCode",
                            NS
                        )
                    )

                    if cpv_elements:

                        cpv_codigo = ", ".join(
                            [
                                el.text.strip()
                                for el in cpv_elements
                                if el.text
                            ]
                        )

                except Exception:
                    pass

                # -------------------------------------------------
                # LUGAR DE EJECUCIÓN
                # -------------------------------------------------

                lugar_ejecucion = "No especificado"

                try:

                    lugar_el = entry.find(
                        ".//cac:ProcurementProject/"
                        "cac:RealizedLocation/"
                        "cbc:CountrySubentity",
                        NS
                    )

                    if (
                        lugar_el is not None
                        and lugar_el.text
                    ):

                        lugar_ejecucion = (
                            lugar_el.text.strip()
                        )

                    else:

                        lugar_el = entry.find(
                            ".//cac:ProcurementProject/"
                            "cac:RealizedLocation/"
                            "cbc:CountrySubentityCode",
                            NS
                        )

                        if (
                            lugar_el is not None
                            and lugar_el.text
                        ):

                            codigo_lugar = (
                                lugar_el.text.strip()
                            )

                            lugar_ejecucion = (
                                MAPEO_NUTS.get(
                                    codigo_lugar,
                                    codigo_lugar
                                )
                            )

                except Exception:
                    pass

                # -------------------------------------------------
                # TIPO DE CONTRATO
                # -------------------------------------------------

                type_code_el = entry.find(
                    ".//cac-place-ext:"
                    "ContractFolderStatus/"
                    "cac:ProcurementProject/"
                    "cbc:TypeCode",
                    NS
                )

                if type_code_el is None:

                    type_code_el = entry.find(
                        ".//cbc:TypeCode",
                        NS
                    )

                tipo_contrato_raw = (
                    type_code_el.text.strip()
                    if (
                        type_code_el is not None
                        and type_code_el.text
                    )
                    else "No especificado"
                )

                tipo_contrato = (
                    traducir_tipo_contrato(
                        tipo_contrato_raw
                    )
                )

                # -------------------------------------------------
                # IMPORTE
                # -------------------------------------------------

                importe = 0.0

                try:

                    presupuesto_el = entry.find(
                        ".//cac:BudgetAmount/"
                        "cbc:EstimatedOverallContractAmount",
                        NS
                    )

                    if presupuesto_el is None:

                        presupuesto_el = entry.find(
                            ".//cac:BudgetAmount/"
                            "cbc:TaxExclusiveAmount",
                            NS
                        )

                    if presupuesto_el is None:

                        presupuesto_el = entry.find(
                            ".//cac:BudgetAmount/"
                            "cbc:TotalAmount",
                            NS
                        )

                    if (
                        presupuesto_el is not None
                        and presupuesto_el.text
                    ):

                        importe = float(
                            presupuesto_el.text
                            .strip()
                            .replace(",", ".")
                        )

                except Exception:
                    pass

                # -------------------------------------------------
                # ÓRGANO
                # -------------------------------------------------

                organo = "Órgano desconocido"

                rutas_organo = [
                    ".//cac-place-ext:"
                    "LocatedContractingParty//"
                    "cac:PartyName//cbc:Name",

                    ".//cac:ContractingParty//"
                    "cac:PartyName//cbc:Name",

                    ".//cac:TenderingParty//"
                    "cac:PartyName//cbc:Name",

                    ".//cac:ContractingParty//"
                    "cac:Party//"
                    "cac:PartyName//cbc:Name",

                    ".//cbc:PartyName//cbc:Name"
                ]

                for ruta in rutas_organo:

                    organo_el = entry.find(
                        ruta,
                        NS
                    )

                    if (
                        organo_el is not None
                        and organo_el.text
                        and organo_el.text.strip()
                    ):

                        organo = (
                            organo_el.text.strip()
                        )

                        break

                # -------------------------------------------------
                # DESCRIPCIÓN
                # -------------------------------------------------

                descripcion = (
                    _texto(
                        entry,
                        ".//cac-place-ext:"
                        "ContractFolderStatus/"
                        "cac:ProcurementProject/"
                        "cbc:Name",
                        NS
                    )
                    or _texto(
                        entry,
                        ".//cac:ProcurementProject/"
                        "cbc:Description",
                        NS
                    )
                    or ""
                )

                # -------------------------------------------------
                # TEXTO PARA EMBEDDING
                # -------------------------------------------------

                texto_evaluacion = (
                    f"passage: Título: {titulo}. "
                    f"Objeto: {descripcion}. "
                    f"Órgano: {organo}. "
                    f"Tipo de contrato: {tipo_contrato}. "
                    f"CPV: {cpv_codigo}. "
                    f"Lugar: {lugar_ejecucion}. "
                    f"Importe: {importe} EUR."
                )

                # -------------------------------------------------
                # DATOS DE LA LICITACIÓN
                # -------------------------------------------------

                licitacion_data = {
                    "enlace": enlace,
                    "titulo": titulo.strip(),
                    "organo": organo.strip(),
                    "fecha": fecha_str,
                    "importe": importe,
                    "cpv": cpv_codigo,
                    "lugar_ejecucion": lugar_ejecucion,
                    "fecha_fin": fecha_fin_str,
                    "tipo_contrato": tipo_contrato,
                    "texto_completo": texto_evaluacion,
                    "fuente": nombre_feed,
                    "_atom_updated": (
                        txt_updated
                        or txt_fecha
                    ),
                }

                # -------------------------------------------------
                # DEDUPLICAR POR EXPEDIENTE
                # -------------------------------------------------

                if (
                    expediente_id
                    not in licitaciones_por_expediente
                    or
                    licitacion_data["_atom_updated"]
                    >
                    licitaciones_por_expediente[
                        expediente_id
                    ]["_atom_updated"]
                ):

                    licitaciones_por_expediente[
                        expediente_id
                    ] = licitacion_data

            # =====================================================
            # SIGUIENTE PÁGINA
            # =====================================================

            next_link_el = root.find(
                "atom:link[@rel='next']",
                NS
            )

            if next_link_el is None:

                next_link_el = root.find(
                    ".//{http://www.w3.org/2005/Atom}"
                    "link[@rel='next']"
                )

            url_actual = (
                next_link_el.get("href")
                if next_link_el is not None
                else None
            )

            if (
                url_actual
                and "contrataciondelestado.es"
                in url_actual
            ):

                url_actual = url_actual.replace(
                    "contrataciondelestado.es",
                    "contrataciondelsectorpublico.gob.es"
                )

            time.sleep(0.5)

        except Exception as e:

            print(
                f"❌ Error de red: {e}. "
                f"Reintentando..."
            )

            time.sleep(5)
            continue

    # ============================================================
    # PREPARAR ENTRADAS
    # ============================================================

    entries_totales = list(
        licitaciones_por_expediente.values()
    )

    for item in entries_totales:
        item.pop(
            "_atom_updated",
            None
        )

    print(
        f"📊 Entradas únicas obtenidas del feed: "
        f"{len(entries_totales)}"
    )

    # ============================================================
    # 4. CARGA GLOBAL Y VALIDACIÓN EN SUPABASE
    # ============================================================

    print(
        "Cargando registros existentes desde "
        "Supabase para validación global..."
    )

    try:

        existentes_resp = (
            supabase.table("licitaciones")
            .select(
                "id,enlace,titulo,organo,fuente,"
                "tipo_contrato,fecha_fin,importe"
            )
            .execute()
        )

        datos_existentes = (
            existentes_resp.data or []
        )

        registros_db = {
            item["enlace"]: item
            for item in datos_existentes
            if "enlace" in item
        }

        # --------------------------------------------------------
        # Índice para detectar duplicados por título + órgano
        # --------------------------------------------------------

        registros_existentes = set()

        for item in datos_existentes:

            t = str(
                item.get(
                    "titulo",
                    ""
                )
            ).strip().lower()

            o_base = normalizar_organo(
                item.get(
                    "organo",
                    ""
                )
            )

            if t or o_base:

                registros_existentes.add(
                    (
                        t,
                        o_base
                    )
                )

        print(
            f"Registros totales cargados desde "
            f"Supabase: {len(registros_db)}"
        )

    except Exception as e:

        print(
            f"Error conectando con Supabase "
            f"para lectura: {e}"
        )

        return

    # ============================================================
    # CONTADORES
    # ============================================================

    licitaciones_nuevas = []

    enlaces_procesados_sesion = set()

    contador_nuevas = 0
    contador_actualizadas = 0
    contador_sin_cambios = 0
    contador_duplicados = 0

    print(
        "Procesando duplicados y preparando sincronización..."
    )

    # ============================================================
    # PROCESAR ENTRADAS
    # ============================================================

    for entry in entries_totales:

        enlace = entry["enlace"]

        # Evitar duplicados dentro de la propia ejecución
        if enlace in enlaces_procesados_sesion:
            continue

        enlaces_procesados_sesion.add(enlace)

        titulo_str = entry["titulo"]
        organo = entry["organo"]
        organo_base = normalizar_organo(organo)

        tipo_contrato = entry["tipo_contrato"]
        importe = entry["importe"]
        fecha_fin_str = entry["fecha_fin"]

        clave_duplicado = (
            titulo_str.strip().lower(),
            organo_base
        )

        # ========================================================
        # CASO 1: REGISTRO YA EXISTENTE POR ENLACE
        # ========================================================

        if enlace in registros_db:

            reg_antiguo = registros_db[enlace]

            fuente_actual = str(
                reg_antiguo.get(
                    "fuente",
                    ""
                )
            )

            tipo_actual = reg_antiguo.get(
                "tipo_contrato",
                ""
            )

            actualizar_datos = {}

            # ----------------------------------------------------
            # Añadir la fuente si todavía no está
            # ----------------------------------------------------

            if not contiene_fuente(
                fuente_actual,
                nombre_feed
            ):

                nueva_fuente = añadir_fuente(
                    fuente_actual,
                    nombre_feed
                )

                actualizar_datos[
                    "fuente"
                ] = nueva_fuente

            # ----------------------------------------------------
            # Completar tipo de contrato si falta
            # ----------------------------------------------------

            if (
                (
                    not tipo_actual
                    or tipo_actual == "No especificado"
                )
                and
                tipo_contrato != "No especificado"
            ):

                actualizar_datos[
                    "tipo_contrato"
                ] = tipo_contrato

            # ----------------------------------------------------
            # Detectar cambios importantes
            #
            # Estos sí provocan es_actualizada=True
            # ----------------------------------------------------

            es_actualizado = (
                reg_antiguo.get("titulo")
                != titulo_str.strip()
                or
                reg_antiguo.get("importe")
                != importe
                or
                reg_antiguo.get("fecha_fin")
                != fecha_fin_str
            )

            if es_actualizado:

                actualizar_datos[
                    "es_actualizada"
                ] = True

            # ----------------------------------------------------
            # Actualizar solo si realmente hay algo que cambiar
            # ----------------------------------------------------

            if actualizar_datos:

                try:

                    supabase.table(
                        "licitaciones"
                    ).update(
                        actualizar_datos
                    ).eq(
                        "enlace",
                        enlace
                    ).execute()

                    reg_antiguo.update(
                        actualizar_datos
                    )

                    if es_actualizado:

                        contador_actualizadas += 1

                    else:

                        # Cambio de metadata únicamente:
                        # fuente o tipo de contrato
                        contador_sin_cambios += 1

                except Exception as e:

                    print(
                        f"⚠️ Error actualizando "
                        f"registro existente "
                        f"{enlace}: {e}"
                    )

            else:

                contador_sin_cambios += 1

            # IMPORTANTE:
            # No se calcula embedding para registros existentes.
            # No se inserta ni se hace upsert.
            continue

        # ========================================================
        # CASO 2: DUPLICADO POR TÍTULO + ÓRGANO
        # ========================================================

        if clave_duplicado in registros_existentes:

            contador_duplicados += 1

            print(
                f"   🔁 Duplicado evitado: "
                f"{titulo_str[:80]}"
            )

            # Intentamos añadir la fuente al registro
            # correspondiente si todavía no la tiene.
            for reg in datos_existentes:

                titulo_db = str(
                    reg.get(
                        "titulo",
                        ""
                    )
                ).strip().lower()

                organo_db = normalizar_organo(
                    reg.get(
                        "organo",
                        ""
                    )
                )

                if (
                    titulo_db,
                    organo_db
                ) == clave_duplicado:

                    fuente_db = str(
                        reg.get(
                            "fuente",
                            ""
                        )
                    )

                    if not contiene_fuente(
                        fuente_db,
                        nombre_feed
                    ):

                        nueva_fuente = añadir_fuente(
                            fuente_db,
                            nombre_feed
                        )

                        try:

                            supabase.table(
                                "licitaciones"
                            ).update({
                                "fuente": nueva_fuente
                            }).eq(
                                "id",
                                reg["id"]
                            ).execute()

                        except Exception as e:

                            print(
                                f"⚠️ Error añadiendo "
                                f"fuente al duplicado: {e}"
                            )

                    break

            continue

        # ========================================================
        # CASO 3: REGISTRO REALMENTE NUEVO
        # ========================================================

        print(
            f"   🆕 Nueva licitación: "
            f"{titulo_str[:80]}"
        )

        # El embedding SOLO se calcula para registros nuevos
        embedding = encoder.encode(
            entry["texto_completo"]
        ).tolist()

        entry["embedding"] = embedding

        entry["es_novedad"] = True
        entry["es_actualizada"] = False
        entry["fuente"] = nombre_feed

        licitaciones_nuevas.append(
            entry
        )

        contador_nuevas += 1

        # Añadir a nuestro índice para evitar duplicados
        # dentro de esta misma ejecución
        registros_existentes.add(
            clave_duplicado
        )

    # ============================================================
    # 5. INSERCIÓN DE NUEVOS REGISTROS
    # ============================================================

    if licitaciones_nuevas:

        total_a_subir = len(
            licitaciones_nuevas
        )

        print(
            f"\n🆕 Subiendo un total de "
            f"{total_a_subir} licitaciones nuevas "
            f"a Supabase..."
        )

        tamano_lote = 5
        subidas_exitosas = 0
        max_intentos = 3

        for i in range(
            0,
            total_a_subir,
            tamano_lote
        ):

            lote = licitaciones_nuevas[
                i:i + tamano_lote
            ]

            num_lote = (
                i // tamano_lote
            ) + 1

            exito = False

            for intento in range(
                1,
                max_intentos + 1
            ):

                try:

                    # IMPORTANTE:
                    # INSERT, no UPSERT.
                    #
                    # Así un registro existente nunca
                    # se sobrescribe accidentalmente.

                    supabase.table(
                        "licitaciones"
                    ).insert(
                        lote
                    ).execute()

                    subidas_exitosas += len(
                        lote
                    )

                    print(
                        f"Progreso ({nombre_feed}): "
                        f"{subidas_exitosas}/"
                        f"{total_a_subir} nuevas "
                        f"insertadas..."
                    )

                    exito = True
                    break

                except Exception as e:

                    print(
                        f"⚠️ Intento "
                        f"{intento}/{max_intentos} "
                        f"fallido para lote "
                        f"{num_lote}: {e}"
                    )

                    if intento < max_intentos:

                        time.sleep(
                            2 * intento
                        )

                    else:

                        print(
                            f"❌ Error definitivo "
                            f"al subir lote "
                            f"{num_lote}."
                        )

            if not exito:
                pass

    else:

        print(
            f"\nℹ️ No hay nuevas licitaciones "
            f"para insertar en {nombre_feed}."
        )

    # ============================================================
    # RESUMEN DEL FEED
    # ============================================================

    print(
        f"\n=================================================="
    )

    print(
        f"📊 RESUMEN {nombre_feed}"
    )

    print(
        f"   🆕 Nuevas: {contador_nuevas}"
    )

    print(
        f"   🔄 Actualizadas: {contador_actualizadas}"
    )

    print(
        f"   ⚪ Sin cambios: {contador_sin_cambios}"
    )

    print(
        f"   🔁 Duplicados evitados: "
        f"{contador_duplicados}"
    )

    print(
        f"   📥 Nuevas insertadas correctamente: "
        f"{subidas_exitosas if licitaciones_nuevas else 0}"
    )

    print(
        f"=================================================="
    )


# ============================================================
# 6. ORQUESTACIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    limpiar_licitaciones_caducadas()

    for feed in FEEDS_ATOM:

        procesar_y_sincronizar_feed(
            feed["nombre"],
            feed["url"]
        )
