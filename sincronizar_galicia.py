from datetime import datetime, date, timedelta
import os
import time
import re
import json
import html
import unicodedata

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client


# ============================================================
# CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print(
    "Cargando modelo de IA (multilingual-e5-small)..."
)

encoder = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu"
)

URL_LISTADO_GALICIA = (
    "https://www.contratosdegalicia.gal/"
    "ultimasPublicaciones.jsp?lang=gl"
)

BASE_URL_GALICIA = (
    "https://www.contratosdegalicia.gal"
)

FUENTE_GALICIA = "Galicia"


# ============================================================
# SESIÓN HTTP
# ============================================================

session = requests.Session()

retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[
        429,
        500,
        502,
        503,
        504
    ],
    allowed_methods=["GET"]
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retries)
)

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,"
        "*/*;q=0.8"
    ),
    "Accept-Language": (
        "gl-ES,gl;q=0.9,es;q=0.8,en;q=0.7"
    )
})


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_texto(texto):
    """Limpia espacios y entidades HTML."""

    if texto is None:
        return ""

    texto = html.unescape(
        str(texto)
    )

    texto = texto.replace(
        "\xa0",
        " "
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


def normalizar_organo(texto):
    """Normaliza el órgano para detectar duplicados."""

    if not texto:
        return ""

    texto = str(
        texto
    ).lower().strip()

    # Eliminar acentos
    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        c
        for c in texto
        if unicodedata.category(c) != "Mn"
    )

    # Eliminar caracteres especiales
    texto = re.sub(
        r"[^a-z0-9\s]",
        "",
        texto
    )

    # Normalizar espacios
    return re.sub(
        r"\s+",
        " ",
        texto
    ).strip()


def normalizar_fuentes(fuente):
    """Convierte la cadena de fuentes en una lista limpia."""

    if not fuente:
        return []

    return [
        f.strip()
        for f in str(fuente).split(",")
        if f.strip()
    ]


def contiene_fuente(
    fuente_actual,
    nombre_fuente
):
    """Comprueba si una fuente concreta está presente."""

    fuentes = normalizar_fuentes(
        fuente_actual
    )

    return any(
        f.casefold() == nombre_fuente.casefold()
        for f in fuentes
    )


def añadir_fuente(
    fuente_actual,
    nombre_fuente
):
    """Añade una fuente sin duplicarla."""

    fuentes = normalizar_fuentes(
        fuente_actual
    )

    if not any(
        f.casefold() == nombre_fuente.casefold()
        for f in fuentes
    ):
        fuentes.append(
            nombre_fuente
        )

    return ", ".join(
        fuentes
    )


def quitar_fuente(
    fuente_actual,
    nombre_fuente
):
    """Elimina únicamente una fuente concreta."""

    fuentes = normalizar_fuentes(
        fuente_actual
    )

    fuentes = [
        f
        for f in fuentes
        if f.casefold() != nombre_fuente.casefold()
    ]

    return ", ".join(
        fuentes
    )


def limpiar_cpv(cpv_raw):
    """Extrae los primeros 8 dígitos del CPV."""

    if not cpv_raw:
        return "No especificado"

    cpv_str = normalizar_texto(
        cpv_raw
    )

    if not cpv_str:
        return "No especificado"

    match = re.search(
        r"(\d{8}(?:-\d)?)",
        cpv_str
    )

    if match:
        return match.group(1)

    return cpv_str[:20]


def parsear_importe(valor):
    """
    Convierte importes gallegos a float.

    Ejemplos:
        15.000,00
        2.965906325E7
        0,00
        15000
    """

    if valor is None:
        return 0.0

    texto = normalizar_texto(
        valor
    )

    if not texto:
        return 0.0

    texto = (
        texto
        .replace("€", "")
        .replace("EUR", "")
        .strip()
    )

    # Notación científica
    if re.search(
        r"[eE][+-]?\d+",
        texto
    ):
        try:
            return float(
                texto.replace(",", ".")
            )
        except (
            ValueError,
            TypeError
        ):
            return 0.0

    texto = texto.replace(
        " ",
        ""
    )

    # Formato europeo:
    # 15.000,00 -> 15000.00
    if "," in texto:

        texto = texto.replace(
            ".",
            ""
        )

        texto = texto.replace(
            ",",
            "."
        )

    else:

        partes = texto.split(".")

        if len(partes) > 2:
            texto = "".join(
                partes
            )

    try:

        return float(
            texto
        )

    except (
        ValueError,
        TypeError
    ):

        return 0.0


def parsear_fecha(
    valor
):
    """Convierte una fecha de Galicia a date."""

    if valor is None:
        return None

    texto = normalizar_texto(
        valor
    )

    if not texto:
        return None

    formatos = [
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S"
    ]

    for formato in formatos:

        try:

            return datetime.strptime(
                texto,
                formato
            )

        except ValueError:
            continue

    return None


def parsear_fecha_publicacion(
    valor
):
    """Devuelve la fecha de publicación en formato YYYY-MM-DD."""

    fecha = parsear_fecha(
        valor
    )

    if not fecha:
        return None

    return fecha.strftime(
        "%Y-%m-%d"
    )


def parsear_fecha_fin(
    valor
):
    """Devuelve la fecha límite en formato YYYY-MM-DD."""

    fecha = parsear_fecha(
        valor
    )

    if not fecha:
        return "No especificada"

    return fecha.strftime(
        "%Y-%m-%d"
    )


# ============================================================
# EXTRAER CAMPOS DEL DETALLE
# ============================================================

def extraer_dt_dd(
    soup,
    etiqueta
):
    """
    Busca una estructura:

        <dt>Etiqueta</dt>
        <dd>Valor</dd>
    """

    etiqueta_normalizada = normalizar_texto(
        etiqueta
    ).lower().rstrip(":")

    for dt in soup.find_all("dt"):

        texto_dt = normalizar_texto(
            dt.get_text(
                " ",
                strip=True
            )
        ).lower().rstrip(":")

        if texto_dt == etiqueta_normalizada:

            dd = dt.find_next_sibling(
                "dd"
            )

            if dd:

                return normalizar_texto(
                    dd.get_text(
                        " ",
                        strip=True
                    )
                )

    return ""


def extraer_estado(
    soup
):
    """Extrae el estado del procedimiento."""

    for p in soup.find_all("p"):

        texto = normalizar_texto(
            p.get_text(
                " ",
                strip=True
            )
        )

        if (
            "Estado do procedemento"
            in texto
        ):

            em = p.find(
                "em"
            )

            if em:

                return normalizar_texto(
                    em.get_text(
                        " ",
                        strip=True
                    )
                )

            texto = re.sub(
                r"Estado do procedemento\s*:?",
                "",
                texto,
                flags=re.IGNORECASE
            )

            return normalizar_texto(
                texto
            )

    return ""


def extraer_organo(
    soup
):
    """Extrae el órgano de contratación."""

    organismo = soup.select_one(
        ".organismo .logo-texto"
    )

    if organismo:

        enlace = organismo.find(
            "a"
        )

        if enlace:

            return normalizar_texto(
                enlace.get_text(
                    " ",
                    strip=True
                )
            )

        return normalizar_texto(
            organismo.get_text(
                " ",
                strip=True
            )
        )

    return ""


def extraer_cpv(
    soup
):
    """Extrae los códigos CPV."""

    panel = soup.find(
        id="collapseCPV"
    )

    if not panel:
        return "No especificado"

    cpvs = []

    for td in panel.find_all(
        "td"
    ):

        texto = normalizar_texto(
            td.get_text(
                " ",
                strip=True
            )
        )

        match = re.search(
            r"\b(\d{8})\b",
            texto
        )

        if match:

            codigo = match.group(
                1
            )

            if codigo not in cpvs:

                cpvs.append(
                    codigo
                )

    if not cpvs:
        return "No especificado"

    return ", ".join(
        cpvs
    )


def extraer_nuts(
    soup
):
    """Extrae las localizaciones NUT."""

    panel = soup.find(
        id="collapseNUT"
    )

    if not panel:
        return "No especificado"

    nuts = []

    for tr in panel.find_all(
        "tr"
    ):

        celdas = tr.find_all(
            "td"
        )

        if not celdas:
            continue

        valores = []

        for td in celdas:

            valor = normalizar_texto(
                td.get_text(
                    " ",
                    strip=True
                )
            )

            if valor:
                valores.append(
                    valor
                )

        if valores:

            texto = " ".join(
                valores
            )

            if texto not in nuts:

                nuts.append(
                    texto
                )

    if not nuts:
        return "No especificado"

    return ", ".join(
        nuts
    )


def construir_enlace_galicia(
    codigo
):
    """Construye el enlace canónico de la licitación."""

    return (
        f"{BASE_URL_GALICIA}/"
        f"licitacion?N={codigo}"
    )


# ============================================================
# DESCARGAR DETALLE DE UNA LICITACIÓN
# ============================================================

def extraer_detalle_galicia(
    codigo,
    fecha_publicacion_lista=None
):
    """Descarga y procesa el detalle de una licitación."""

    enlace = construir_enlace_galicia(
        codigo
    )

    try:

        response = session.get(
            enlace,
            timeout=30
        )

        response.raise_for_status()

    except Exception as e:

        print(
            f"  -> Error descargando detalle "
            f"{codigo}: {e}"
        )

        return None

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    # --------------------------------------------------------
    # ÓRGANO
    # --------------------------------------------------------

    organo = extraer_organo(
        soup
    )

    # --------------------------------------------------------
    # TÍTULO
    # --------------------------------------------------------

    titulo = extraer_dt_dd(
        soup,
        "Obxecto"
    )

    # --------------------------------------------------------
    # TIPO DE CONTRATO
    # --------------------------------------------------------

    tipo_contrato = extraer_dt_dd(
        soup,
        "Tipo de contrato"
    )

    if not tipo_contrato:

        tipo_contrato = (
            "No especificado"
        )

    # --------------------------------------------------------
    # PRESUPUESTO BASE
    # --------------------------------------------------------

    presupuesto_texto = extraer_dt_dd(
        soup,
        "Orzamento base de licitación"
    )

    importe = parsear_importe(
        presupuesto_texto
    )

    # --------------------------------------------------------
    # VALOR ESTIMADO
    # --------------------------------------------------------

    valor_estimado_texto = extraer_dt_dd(
        soup,
        "Valor estimado"
    )

    valor_estimado = parsear_importe(
        valor_estimado_texto
    )

    # Si el presupuesto base no existe,
    # usamos el valor estimado como respaldo.
    if (
        importe == 0.0
        and valor_estimado > 0
    ):

        importe = valor_estimado

    # --------------------------------------------------------
    # FECHA DE PUBLICACIÓN
    # --------------------------------------------------------

    fecha_publicacion_texto = extraer_dt_dd(
        soup,
        "Data de difusión en CPG"
    )

    if not fecha_publicacion_texto:

        fecha_publicacion_texto = extraer_dt_dd(
            soup,
            "Data de difusión en Contratos Públicos de Galicia"
        )

    fecha_publicacion = (
        parsear_fecha_publicacion(
            fecha_publicacion_texto
        )
    )

    # Si no se encuentra en el detalle,
    # utilizamos la fecha del listado.
    if not fecha_publicacion:

        fecha_publicacion = (
            parsear_fecha_publicacion(
                fecha_publicacion_lista
            )
        )

    # --------------------------------------------------------
    # FECHA FIN
    # --------------------------------------------------------

    fecha_fin_texto = extraer_dt_dd(
        soup,
        "Data e hora límite"
    )

    fecha_fin = parsear_fecha_fin(
        fecha_fin_texto
    )

    # --------------------------------------------------------
    # CPV
    # --------------------------------------------------------

    cpv = extraer_cpv(
        soup
    )

    # --------------------------------------------------------
    # LUGAR / NUT
    # --------------------------------------------------------

    lugar_ejecucion = extraer_nuts(
        soup
    )

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

    estado = extraer_estado(
        soup
    )

    # --------------------------------------------------------
    # DATOS
    # --------------------------------------------------------

    return {
        "enlace": enlace,
        "titulo": titulo,
        "organo": organo,
        "fecha": fecha_publicacion,
        "importe": importe,
        "tipo_contrato": tipo_contrato,
        "cpv": cpv,
        "lugar_ejecucion": lugar_ejecucion,
        "fecha_fin": fecha_fin,
        "estado": estado
    }


# ============================================================
# DESCARGAR LISTADO DE GALICIA
# ============================================================

def obtener_registros_galicia():
    """
    Obtiene el JSON completo de resSearch.
    """

    print(
        "\nConsultando publicaciones de Galicia..."
    )

    try:

        response = session.get(
            URL_LISTADO_GALICIA,
            timeout=30
        )

        response.raise_for_status()

    except Exception as e:

        print(
            f"Error conectando con Galicia: {e}"
        )

        return []

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    input_res_search = soup.find(
        "input",
        id="resSearch"
    )

    if not input_res_search:

        print(
            "No se encontró el campo resSearch."
        )

        return []

    raw = input_res_search.get(
        "value",
        ""
    )

    if not raw:

        print(
            "El campo resSearch está vacío."
        )

        return []

    try:

        raw = html.unescape(
            raw
        )

        registros = json.loads(
            raw
        )

    except Exception as e:

        print(
            f"Error interpretando resSearch: {e}"
        )

        return []

    if not isinstance(
        registros,
        list
    ):

        print(
            "El contenido de resSearch "
            "no es una lista."
        )

        return []

    print(
        f"Total registros obtenidos del listado Galicia: "
        f"{len(registros)}"
    )

    return registros


# ============================================================
# ESTADOS A EXCLUIR
# ============================================================

def estado_no_valido(
    estado
):
    """Comprueba si el procedimiento está cerrado."""

    if not estado:
        return False

    estado_limpio = normalizar_texto(
        estado
    ).lower()

    estados_excluidos = [
        "resuelta",
        "cerrada",
        "adjudicada",
        "anulada",
        "desistida",
        "renunciada",
        "cancelada"
    ]

    return any(
        estado_excluido in estado_limpio
        for estado_excluido in estados_excluidos
    )


# ============================================================
# TEXTO PARA EMBEDDING
# ============================================================

def construir_texto_completo(
    titulo,
    organo,
    tipo_contrato,
    lugar_ejecucion,
    importe,
    cpv
):
    """Construye el texto utilizado para el embedding."""

    return (
        f"passage: Título: {titulo}. "
        f"Órgano: {organo}. "
        f"Tipo Contrato: {tipo_contrato}. "
        f"Lugar: {lugar_ejecucion}. "
        f"Importe: {importe} EUR. "
        f"CPV: {cpv}."
    )


# ============================================================
# SINCRONIZACIÓN GALICIA
# ============================================================

def sincronizar_licitaciones_galicia():

    hoy_date = datetime.now().date()

    limite_fecha = (
        hoy_date
        - timedelta(days=2)
    )

    print(
        "============================================================"
    )

    print(
        "SINCRONIZACIÓN LICITACIONES GALICIA"
    )

    print(
        "============================================================"
    )

    print(
        f"Fecha actual: {hoy_date}"
    )

    print(
        f"Procesando publicaciones desde "
        f"{limite_fecha} hasta {hoy_date}"
    )

    # ========================================================
    # 1. DESCARGAR DATOS DE GALICIA
    # ========================================================

    results = obtener_registros_galicia()

    if not results:

        print(
            "No se han obtenido registros de Galicia."
        )

        return

    # ========================================================
    # 2. CARGAR REGISTROS EXISTENTES DE SUPABASE
    # ========================================================

    try:

        existentes_resp = (
            supabase
            .table("licitaciones")
            .select(
                "id, enlace, titulo, organo, fuente, fecha, "
                "importe, tipo_contrato, cpv, fecha_fin, "
                "lugar_ejecucion, es_novedad, es_actualizada"
            )
            .execute()
        )

        registros_db = {}
        registros_existentes = set()
        ids_flags_galicia = []

        for item in (
            existentes_resp.data or []
        ):

            # ----------------------------------------------
            # Mapa por enlace
            # ----------------------------------------------

            enlace_item = item.get(
                "enlace"
            )

            if enlace_item:

                registros_db[
                    enlace_item
                ] = item

            # ----------------------------------------------
            # Mapa por título + órgano
            # ----------------------------------------------

            titulo_item = str(
                item.get("titulo") or ""
            ).strip().lower()

            organo_item = normalizar_organo(
                item.get(
                    "organo",
                    ""
                )
            )

            if titulo_item or organo_item:

                registros_existentes.add(
                    (
                        titulo_item,
                        organo_item
                    )
                )

            # ----------------------------------------------
            # Registros de Galicia con flags anteriores
            # ----------------------------------------------

            fuente_item = str(
                item.get("fuente") or ""
            )

            if (
                contiene_fuente(
                    fuente_item,
                    FUENTE_GALICIA
                )
                and (
                    item.get(
                        "es_novedad"
                    ) is True
                    or
                    item.get(
                        "es_actualizada"
                    ) is True
                )
            ):

                ids_flags_galicia.append(
                    item["id"]
                )

        print(
            "Registros cargados desde Supabase para "
            f"validación: {len(existentes_resp.data or [])}"
        )

    except Exception as e:

        print(
            f"Error conectando con Supabase para lectura: {e}"
        )

        return

    # ========================================================
    # 2.1. RESETEAR ETIQUETAS ANTERIORES DE GALICIA
    # ========================================================

    if ids_flags_galicia:

        print(
            f"Reseteando etiquetas anteriores de "
            f"{len(ids_flags_galicia)} registros de Galicia..."
        )

        tamano_reset = 25
        max_intentos_reset = 3

        reset_correcto = True
        total_reseteadas = 0

        for i in range(
            0,
            len(ids_flags_galicia),
            tamano_reset
        ):

            lote_ids = ids_flags_galicia[
                i:i + tamano_reset
            ]

            num_lote_reset = (
                i // tamano_reset
            ) + 1

            exito_lote = False

            for intento in range(
                1,
                max_intentos_reset + 1
            ):

                try:

                    (
                        supabase
                        .table("licitaciones")
                        .update({
                            "es_novedad": False,
                            "es_actualizada": False
                        })
                        .in_(
                            "id",
                            lote_ids
                        )
                        .execute()
                    )

                    total_reseteadas += len(
                        lote_ids
                    )

                    print(
                        f"  -> Lote de etiquetas "
                        f"{num_lote_reset} reseteado con éxito "
                        f"({len(lote_ids)} registros)."
                    )

                    exito_lote = True

                    break

                except Exception as e:

                    print(
                        f"  -> Intento "
                        f"{intento}/{max_intentos_reset} "
                        f"fallido para lote de etiquetas "
                        f"{num_lote_reset}: {e}"
                    )

                    if intento < max_intentos_reset:

                        time.sleep(
                            2 * intento
                        )

                    else:

                        print(
                            f"  -> Error definitivo al "
                            f"resetear el lote de etiquetas "
                            f"{num_lote_reset}."
                        )

                        reset_correcto = False

            if not exito_lote:
                continue

        if reset_correcto:

            print(
                f"Etiquetas anteriores reseteadas "
                f"correctamente: {total_reseteadas} registros."
            )

        else:

            print(
                "Aviso: no se pudieron resetear "
                "todas las etiquetas anteriores."
            )

    else:

        print(
            "No hay etiquetas anteriores de Galicia "
            "que resetear."
        )

    # ========================================================
    # 3. PROCESAR LICITACIONES
    # ========================================================

    licitaciones_validas = []

    enlaces_procesados_sesion = set()

    claves_procesadas_sesion = set()

    nuevas = 0
    actualizadas = 0
    duplicadas = 0
    errores = 0
    descartadas_estado = 0

    for i, aviso in enumerate(
        results,
        1
    ):

        codigo = aviso.get(
            "codigo"
        )

        if not codigo:
            continue

        # ----------------------------------------------------
        # FECHA DEL LISTADO
        # ----------------------------------------------------

        fecha_pub_str = aviso.get(
            "fechaPublicacion",
            ""
        )

        fecha_pub = parsear_fecha(
            fecha_pub_str
        )

        if not fecha_pub:
            continue

        pub_date = fecha_pub.date()

        if (
            pub_date < limite_fecha
            or pub_date > hoy_date
        ):
            continue

        fecha_pub_formateada = (
            pub_date.strftime(
                "%Y-%m-%d"
            )
        )

        # ----------------------------------------------------
        # ENLACE
        # ----------------------------------------------------

        enlace = construir_enlace_galicia(
            codigo
        )

        if (
            enlace in enlaces_procesados_sesion
        ):
            continue

        enlaces_procesados_sesion.add(
            enlace
        )

        print(
            f"\n[{i}/{len(results)}] "
            f"Procesando licitación {codigo}..."
        )

        # ----------------------------------------------------
        # OBTENER DETALLE
        # ----------------------------------------------------

        detalle = extraer_detalle_galicia(
            codigo,
            fecha_pub_str
        )

        if not detalle:

            errores += 1

            continue

        # ----------------------------------------------------
        # CAMPOS
        # ----------------------------------------------------

        titulo = normalizar_texto(
            detalle.get(
                "titulo"
            )
        )

        if not titulo:

            titulo = normalizar_texto(
                aviso.get(
                    "asunto"
                )
            )

        organo = normalizar_texto(
            detalle.get(
                "organo"
            )
        )

        if not organo:

            organo = normalizar_texto(
                aviso.get(
                    "org_desc"
                )
            )

        tipo_contrato = normalizar_texto(
            detalle.get(
                "tipo_contrato"
            )
        )

        if not tipo_contrato:

            tipo_contrato = (
                "No especificado"
            )

        importe = detalle.get(
            "importe",
            0.0
        )

        cpv = detalle.get(
            "cpv",
            "No especificado"
        )

        lugar_ejecucion = detalle.get(
            "lugar_ejecucion",
            "No especificado"
        )

        fecha_fin = detalle.get(
            "fecha_fin",
            "No especificada"
        )

        estado = detalle.get(
            "estado",
            ""
        )

        # ----------------------------------------------------
        # FECHA DE PUBLICACIÓN
        # ----------------------------------------------------

        fecha = detalle.get(
            "fecha"
        )

        if not fecha:

            fecha = fecha_pub_formateada

        # ----------------------------------------------------
        # ESTADO
        # ----------------------------------------------------

        if estado_no_valido(
            estado
        ):

            print(
                f"  -> Descartada por estado: "
                f"{estado}"
            )

            descartadas_estado += 1

            continue

        # ----------------------------------------------------
        # CLAVE DE DUPLICADO
        # ----------------------------------------------------

        titulo_normalizado = (
            titulo
            .lower()
            .strip()
        )

        organo_normalizado = (
            normalizar_organo(
                organo
            )
        )

        clave_duplicado = (
            titulo_normalizado,
            organo_normalizado
        )

        # ====================================================
        # 3.1. EXISTE POR ENLACE
        # ====================================================

        if enlace in registros_db:

            reg_antiguo = registros_db[
                enlace
            ]

            fuente_actual = str(
                reg_antiguo.get(
                    "fuente"
                ) or ""
            )

            actualizar_datos = {}

            # ----------------------------------------------
            # Añadir Galicia como fuente
            # ----------------------------------------------

            if not contiene_fuente(
                fuente_actual,
                FUENTE_GALICIA
            ):

                actualizar_datos[
                    "fuente"
                ] = añadir_fuente(
                    fuente_actual,
                    FUENTE_GALICIA
                )

            # ----------------------------------------------
            # Completar tipo de contrato
            # ----------------------------------------------

            tipo_actual = normalizar_texto(
                reg_antiguo.get(
                    "tipo_contrato"
                )
            )

            if (
                not tipo_actual
                or tipo_actual == "No especificado"
            ) and (
                tipo_contrato != "No especificado"
            ):

                actualizar_datos[
                    "tipo_contrato"
                ] = tipo_contrato

            # ----------------------------------------------
            # Detectar cambios reales
            # ----------------------------------------------

            cambios_reales = False

            if (
                normalizar_texto(
                    reg_antiguo.get(
                        "titulo"
                    )
                )
                != titulo
            ):

                actualizar_datos[
                    "titulo"
                ] = titulo

                cambios_reales = True

            if (
                normalizar_texto(
                    reg_antiguo.get(
                        "organo"
                    )
                )
                != organo
            ):

                actualizar_datos[
                    "organo"
                ] = organo

                cambios_reales = True

            if (
                normalizar_texto(
                    reg_antiguo.get(
                        "fecha"
                    )
                )
                != normalizar_texto(
                    fecha
                )
            ):

                actualizar_datos[
                    "fecha"
                ] = fecha

                cambios_reales = True

            try:

                importe_antiguo = float(
                    reg_antiguo.get(
                        "importe"
                    ) or 0.0
                )

            except (
                ValueError,
                TypeError
            ):

                importe_antiguo = 0.0

            if abs(
                importe_antiguo
                - importe
            ) > 0.01:

                actualizar_datos[
                    "importe"
                ] = importe

                cambios_reales = True

            if (
                tipo_contrato
                and tipo_contrato
                != "No especificado"
                and tipo_contrato
                != tipo_actual
            ):

                actualizar_datos[
                    "tipo_contrato"
                ] = tipo_contrato

                cambios_reales = True

            cpv_actual = normalizar_texto(
                reg_antiguo.get(
                    "cpv"
                )
            )

            if (
                cpv
                and cpv != "No especificado"
                and cpv != cpv_actual
            ):

                actualizar_datos[
                    "cpv"
                ] = cpv

                cambios_reales = True

            lugar_actual = normalizar_texto(
                reg_antiguo.get(
                    "lugar_ejecucion"
                )
            )

            if (
                lugar_ejecucion
                and lugar_ejecucion
                != "No especificado"
                and lugar_ejecucion
                != lugar_actual
            ):

                actualizar_datos[
                    "lugar_ejecucion"
                ] = lugar_ejecucion

                cambios_reales = True

            fecha_fin_actual = normalizar_texto(
                reg_antiguo.get(
                    "fecha_fin"
                )
            )

            if (
                fecha_fin
                and fecha_fin
                != "No especificada"
                and fecha_fin
                != fecha_fin_actual
            ):

                actualizar_datos[
                    "fecha_fin"
                ] = fecha_fin

                cambios_reales = True

            # ----------------------------------------------
            # Marcar actualización
            # ----------------------------------------------

            actualizar_datos[
                "es_novedad"
            ] = False

            actualizar_datos[
                "es_actualizada"
            ] = cambios_reales

            # ----------------------------------------------
            # Regenerar embedding si cambió información
            # ----------------------------------------------

            if cambios_reales:

                titulo_embedding = actualizar_datos.get(
                    "titulo",
                    reg_antiguo.get(
                        "titulo"
                    )
                )

                organo_embedding = actualizar_datos.get(
                    "organo",
                    reg_antiguo.get(
                        "organo"
                    )
                )

                importe_embedding = actualizar_datos.get(
                    "importe",
                    reg_antiguo.get(
                        "importe"
                    )
                )

                tipo_embedding = actualizar_datos.get(
                    "tipo_contrato",
                    reg_antiguo.get(
                        "tipo_contrato"
                    )
                )

                cpv_embedding = actualizar_datos.get(
                    "cpv",
                    reg_antiguo.get(
                        "cpv"
                    )
                )

                lugar_embedding = actualizar_datos.get(
                    "lugar_ejecucion",
                    reg_antiguo.get(
                        "lugar_ejecucion"
                    )
                )

                texto_completo = (
                    construir_texto_completo(
                        titulo_embedding,
                        organo_embedding,
                        tipo_embedding,
                        lugar_embedding,
                        importe_embedding,
                        cpv_embedding
                    )
                )

                try:

                    embedding = encoder.encode(
                        texto_completo
                    ).tolist()

                    actualizar_datos[
                        "texto_completo"
                    ] = texto_completo

                    actualizar_datos[
                        "embedding"
                    ] = embedding

                except Exception as e:

                    print(
                        f"  -> Error generando embedding "
                        f"de actualización: {e}"
                    )

                actualizadas += 1

                print(
                    "  -> Licitación actualizada."
                )

            elif "fuente" in actualizar_datos:

                print(
                    "  -> Añadida Galicia como fuente."
                )

            else:

                print(
                    "  -> Sin cambios."
                )

            # ----------------------------------------------
            # Actualizar BBDD
            # ----------------------------------------------

            if actualizar_datos:

                try:

                    (
                        supabase
                        .table("licitaciones")
                        .update(
                            actualizar_datos
                        )
                        .eq(
                            "enlace",
                            enlace
                        )
                        .execute()
                    )

                    reg_antiguo.update(
                        actualizar_datos
                    )

                except Exception as e:

                    print(
                        f"  -> Error actualizando "
                        f"registro existente: {e}"
                    )

                    errores += 1

            claves_procesadas_sesion.add(
                clave_duplicado
            )

            continue

        # ====================================================
        # 3.2. EXISTE POR TÍTULO + ÓRGANO
        # ====================================================

        if clave_duplicado in registros_existentes:

            registro_duplicado = None

            for reg in registros_db.values():

                titulo_db = str(
                    reg.get(
                        "titulo"
                    ) or ""
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

                    registro_duplicado = reg

                    break

            if registro_duplicado:

                fuente_actual = str(
                    registro_duplicado.get(
                        "fuente"
                    ) or ""
                )

                if not contiene_fuente(
                    fuente_actual,
                    FUENTE_GALICIA
                ):

                    nueva_fuente = añadir_fuente(
                        fuente_actual,
                        FUENTE_GALICIA
                    )

                    try:

                        (
                            supabase
                            .table("licitaciones")
                            .update({
                                "fuente": nueva_fuente
                            })
                            .eq(
                                "id",
                                registro_duplicado[
                                    "id"
                                ]
                            )
                            .execute()
                        )

                        registro_duplicado[
                            "fuente"
                        ] = nueva_fuente

                        print(
                            "  -> Duplicada por título + órgano. "
                            "Añadida Galicia como fuente."
                        )

                    except Exception as e:

                        print(
                            f"  -> Error añadiendo fuente "
                            f"Galicia al duplicado: {e}"
                        )

                        errores += 1

                else:

                    print(
                        "  -> Ya existe con Galicia."
                    )

            duplicadas += 1

            claves_procesadas_sesion.add(
                clave_duplicado
            )

            continue

        # ====================================================
        # 3.3. DUPLICADO DENTRO DE ESTA EJECUCIÓN
        # ====================================================

        if clave_duplicado in claves_procesadas_sesion:

            print(
                "  -> Duplicado dentro de la misma ejecución."
            )

            duplicadas += 1

            continue

        # ====================================================
        # 3.4. NUEVA LICITACIÓN
        # ====================================================

        claves_procesadas_sesion.add(
            clave_duplicado
        )

        texto_completo = (
            construir_texto_completo(
                titulo,
                organo,
                tipo_contrato,
                lugar_ejecucion,
                importe,
                cpv
            )
        )

        try:

            embedding = encoder.encode(
                texto_completo
            ).tolist()

        except Exception as e:

            print(
                f"  -> Error generando embedding: {e}"
            )

            errores += 1

            continue

        elemento = {
            "titulo": titulo,
            "organo": organo,
            "fecha": fecha,
            "importe": importe,
            "enlace": enlace,
            "texto_completo": texto_completo,
            "embedding": embedding,
            "fecha_fin": fecha_fin,
            "lugar_ejecucion": lugar_ejecucion,
            "cpv": cpv,
            "tipo_contrato": tipo_contrato,
            "es_novedad": True,
            "es_actualizada": False,
            "fuente": FUENTE_GALICIA
        }

        licitaciones_validas.append(
            elemento
        )

        nuevas += 1

        # ----------------------------------------------------
        # MUY IMPORTANTE:
        # actualizar también los mapas en memoria
        # ----------------------------------------------------

        registros_existentes.add(
            clave_duplicado
        )

        registros_db[
            enlace
        ] = elemento

    # ========================================================
    # 4. LIMPIEZA AUTOMÁTICA DE CADUCADAS
    # ========================================================

    print(
        "\nComprobando licitaciones caducadas de Galicia..."
    )

    try:

        todos_db = (
            supabase
            .table("licitaciones")
            .select(
                "id, enlace, fecha_fin, fuente"
            )
            .ilike(
                "fuente",
                "%Galicia%"
            )
            .execute()
        )

        ids_a_borrar = []
        ids_a_actualizar = []

        for item in (
            todos_db.data or []
        ):

            f_fin = item.get(
                "fecha_fin"
            )

            if (
                not f_fin
                or f_fin == "No especificada"
            ):
                continue

            try:

                f_cierre = datetime.strptime(
                    str(f_fin),
                    "%Y-%m-%d"
                ).date()

            except ValueError:

                continue

            if f_cierre < hoy_date:

                fuente_actual = str(
                    item.get(
                        "fuente"
                    ) or ""
                )

                fuentes = normalizar_fuentes(
                    fuente_actual
                )

                # ------------------------------------------
                # Galicia es la única fuente
                # ------------------------------------------

                if (
                    len(fuentes) == 1
                    and contiene_fuente(
                        fuente_actual,
                        FUENTE_GALICIA
                    )
                ):

                    ids_a_borrar.append(
                        item["id"]
                    )

                # ------------------------------------------
                # Hay otras fuentes
                # ------------------------------------------

                elif contiene_fuente(
                    fuente_actual,
                    FUENTE_GALICIA
                ):

                    nueva_fuente = quitar_fuente(
                        fuente_actual,
                        FUENTE_GALICIA
                    )

                    ids_a_actualizar.append(
                        (
                            item["id"],
                            nueva_fuente
                        )
                    )

        # ----------------------------------------------------
        # ELIMINAR REGISTROS
        # ----------------------------------------------------

        if ids_a_borrar:

            for i in range(
                0,
                len(ids_a_borrar),
                50
            ):

                lote_ids = ids_a_borrar[
                    i:i + 50
                ]

                (
                    supabase
                    .table("licitaciones")
                    .delete()
                    .in_(
                        "id",
                        lote_ids
                    )
                    .execute()
                )

            print(
                f"Eliminadas {len(ids_a_borrar)} "
                f"licitaciones caducadas cuya única "
                f"fuente era Galicia."
            )

        # ----------------------------------------------------
        # QUITAR SOLO GALICIA
        # ----------------------------------------------------

        for (
            registro_id,
            nueva_fuente
        ) in ids_a_actualizar:

            try:

                (
                    supabase
                    .table("licitaciones")
                    .update({
                        "fuente": nueva_fuente
                    })
                    .eq(
                        "id",
                        registro_id
                    )
                    .execute()
                )

            except Exception as e:

                print(
                    f"Error quitando fuente Galicia "
                    f"del registro {registro_id}: {e}"
                )

        if ids_a_actualizar:

            print(
                f"Quitada la fuente Galicia de "
                f"{len(ids_a_actualizar)} licitaciones "
                f"caducadas que tenían otras fuentes."
            )

    except Exception as e:

        print(
            f"Error en la limpieza de caducadas: {e}"
        )

    # ========================================================
    # 5. INSERTAR NUEVAS EN SUPABASE
    # ========================================================

    if licitaciones_validas:

        total_a_subir = len(
            licitaciones_validas
        )

        print(
            f"\nSubiendo un total de {total_a_subir} "
            f"licitaciones nuevas a Supabase..."
        )

        tamano_lote = 5
        subidas_exitosas = 0
        max_intentos = 3

        for i in range(
            0,
            total_a_subir,
            tamano_lote
        ):

            lote = licitaciones_validas[
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

                    (
                        supabase
                        .table("licitaciones")
                        .insert(
                            lote
                        )
                        .execute()
                    )

                    subidas_exitosas += len(
                        lote
                    )

                    print(
                        f"Progreso: "
                        f"{subidas_exitosas}/"
                        f"{total_a_subir} "
                        f"licitaciones procesadas..."
                    )

                    exito = True

                    break

                except Exception as e:

                    print(
                        f"Intento "
                        f"{intento}/{max_intentos} "
                        f"fallido para lote Galicia "
                        f"{num_lote}: {e}"
                    )

                    if intento < max_intentos:

                        time.sleep(
                            2 * intento
                        )

                    else:

                        print(
                            f"Error definitivo al "
                            f"subir lote Galicia "
                            f"{num_lote}."
                        )

            if not exito:
                continue

        print(
            f"Sincronización completada. "
            f"Se han insertado {subidas_exitosas} "
            f"de {total_a_subir} licitaciones nuevas."
        )

    else:

        print(
            "\nNo hay licitaciones nuevas para insertar."
        )

    # ========================================================
    # 6. RESUMEN
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        "RESUMEN SINCRONIZACIÓN GALICIA"
    )

    print(
        "============================================================"
    )

    print(
        f"Nuevas:                 {nuevas}"
    )

    print(
        f"Actualizadas:           {actualizadas}"
    )

    print(
        f"Duplicadas:             {duplicadas}"
    )

    print(
        f"Descartadas por estado: {descartadas_estado}"
    )

    print(
        f"Errores:                {errores}"
    )

    print(
        "============================================================"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    sincronizar_licitaciones_galicia()
