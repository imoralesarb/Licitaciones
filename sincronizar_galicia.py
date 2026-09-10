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

URL_LISTADO_GALICIA = (
    "https://www.contratosdegalicia.gal/"
    "ultimasPublicaciones.jsp?lang=gl"
)

BASE_URL_GALICIA = "https://www.contratosdegalicia.gal"

FUENTE_GALICIA = "Galicia"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

BATCH_SIZE = 5

DIAS_ATRAS = 2


# ============================================================
# MODELO DE EMBEDDINGS
# ============================================================

print("Cargando modelo de embeddings...")

modelo = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cpu"
)

print("Modelo cargado.")


# ============================================================
# SESIÓN HTTP
# ============================================================

session = requests.Session()

retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
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
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "gl-ES,gl;q=0.9,es;q=0.8,en;q=0.7"
})


# ============================================================
# FUNCIONES DE TEXTO
# ============================================================

def normalizar_texto(valor):
    """
    Limpia HTML, espacios y entidades.
    """
    if valor is None:
        return ""

    texto = html.unescape(str(valor))

    texto = texto.replace("\xa0", " ")

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def normalizar_organo(org):
    """
    Normaliza el órgano para poder comparar registros
    aunque tengan pequeñas diferencias de formato.
    """
    if not org:
        return ""

    texto = normalizar_texto(org).lower()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


# ============================================================
# FUNCIONES DE FUENTES
# ============================================================

def normalizar_fuentes(fuente):
    """
    Convierte la cadena de fuentes en una lista limpia.
    """
    if not fuente:
        return []

    return [
        f.strip()
        for f in str(fuente).split(",")
        if f.strip()
    ]


def contiene_fuente(fuente, fuente_buscada):
    """
    Comprueba una fuente de forma exacta,
    evitando falsos positivos por substring.
    """
    fuentes = normalizar_fuentes(fuente)

    return any(
        f.lower() == fuente_buscada.lower()
        for f in fuentes
    )


def añadir_fuente(fuente_actual, nueva_fuente):
    """
    Añade una fuente si todavía no está presente.
    """
    fuentes = normalizar_fuentes(fuente_actual)

    if not any(
        f.lower() == nueva_fuente.lower()
        for f in fuentes
    ):
        fuentes.append(nueva_fuente)

    return ", ".join(fuentes)


def quitar_fuente(fuente_actual, fuente_quitar):
    """
    Elimina una fuente concreta.
    """
    fuentes = normalizar_fuentes(fuente_actual)

    fuentes = [
        f for f in fuentes
        if f.lower() != fuente_quitar.lower()
    ]

    return ", ".join(fuentes)


# ============================================================
# PARSEO DE IMPORTES
# ============================================================

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

    texto = normalizar_texto(valor)

    if not texto:
        return 0.0

    texto = texto.replace("€", "")
    texto = texto.replace("EUR", "")
    texto = texto.strip()

    # Notación científica
    if re.search(r"[eE][+-]?\d+", texto):
        try:
            return float(texto.replace(",", "."))
        except Exception:
            return 0.0

    # Eliminar espacios
    texto = texto.replace(" ", "")

    # Formato europeo:
    # 15.000,00 -> 15000.00
    if "," in texto:
        texto = texto.replace(".", "")
        texto = texto.replace(",", ".")

    else:
        # Si solo tiene puntos, puede ser decimal
        # o separador de miles.
        partes = texto.split(".")

        if len(partes) > 2:
            texto = "".join(partes)

    try:
        return float(texto)
    except Exception:
        return 0.0


# ============================================================
# PARSEO DE FECHAS
# ============================================================

def parsear_fecha_publicacion(valor):
    """
    Convierte fechas de publicación de Galicia
    a date.
    """

    if valor is None:
        return None

    texto = normalizar_texto(valor)

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
            ).date()
        except ValueError:
            pass

    return None


def parsear_fecha_fin(valor):
    """
    Convierte la fecha límite de Galicia
    al formato YYYY-MM-DD utilizado para fecha_fin.
    """

    if valor is None:
        return None

    texto = normalizar_texto(valor)

    if not texto:
        return None

    formatos = [
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S"
    ]

    for formato in formatos:
        try:
            fecha = datetime.strptime(
                texto,
                formato
            )

            return fecha.strftime("%Y-%m-%d")

        except ValueError:
            pass

    return None


# ============================================================
# EXTRAER DT/DD DEL DETALLE
# ============================================================

def extraer_dt_dd(soup, etiqueta):
    """
    Busca una estructura:

        <dt>Etiqueta</dt>
        <dd>Valor</dd>

    y devuelve el contenido del dd.
    """

    etiqueta_normalizada = normalizar_texto(
        etiqueta
    ).lower()

    for dt in soup.find_all("dt"):

        texto_dt = normalizar_texto(
            dt.get_text(" ", strip=True)
        ).lower()

        if texto_dt.rstrip(":") == etiqueta_normalizada.rstrip(":"):

            dd = dt.find_next_sibling("dd")

            if dd:
                return normalizar_texto(
                    dd.get_text(" ", strip=True)
                )

    return ""


# ============================================================
# EXTRAER ESTADO
# ============================================================

def extraer_estado(soup):
    """
    Extrae el estado del procedimiento.
    """

    for p in soup.find_all("p"):

        texto = normalizar_texto(
            p.get_text(" ", strip=True)
        )

        if "Estado do procedemento" in texto:

            em = p.find("em")

            if em:
                return normalizar_texto(
                    em.get_text(" ", strip=True)
                )

            texto = re.sub(
                r"Estado do procedemento\s*:?",
                "",
                texto,
                flags=re.IGNORECASE
            )

            return normalizar_texto(texto)

    return ""


# ============================================================
# EXTRAER CPV
# ============================================================

def extraer_cpv(soup):
    """
    Extrae los códigos CPV del panel collapseCPV.
    """

    cpvs = []

    panel = soup.find(
        id="collapseCPV"
    )

    if not panel:
        return ""

    for td in panel.find_all("td"):

        texto = normalizar_texto(
            td.get_text(" ", strip=True)
        )

        coincidencia = re.search(
            r"\b(\d{8})\b",
            texto
        )

        if coincidencia:
            codigo = coincidencia.group(1)

            if codigo not in cpvs:
                cpvs.append(codigo)

    return ", ".join(cpvs)


# ============================================================
# EXTRAER NUT
# ============================================================

def extraer_nuts(soup):
    """
    Extrae las localizaciones NUT.
    """

    nuts = []

    panel = soup.find(
        id="collapseNUT"
    )

    if not panel:
        return ""

    for tr in panel.find_all("tr"):

        celdas = tr.find_all("td")

        if not celdas:
            continue

        valores = [
            normalizar_texto(
                td.get_text(" ", strip=True)
            )
            for td in celdas
        ]

        valores = [
            v for v in valores
            if v
        ]

        if valores:
            texto = " ".join(valores)

            if texto not in nuts:
                nuts.append(texto)

    return ", ".join(nuts)


# ============================================================
# EXTRAER ÓRGANO
# ============================================================

def extraer_organo(soup):
    """
    Extrae el órgano desde:

        div.organismo .logo-texto
    """

    organismo = soup.select_one(
        ".organismo .logo-texto"
    )

    if organismo:

        enlace = organismo.find("a")

        if enlace:
            return normalizar_texto(
                enlace.get_text(" ", strip=True)
            )

        return normalizar_texto(
            organismo.get_text(" ", strip=True)
        )

    return ""


# ============================================================
# EXTRAER ENLACE
# ============================================================

def construir_enlace_galicia(codigo):
    """
    Construye un enlace canónico de procedimiento.
    """

    return (
        f"{BASE_URL_GALICIA}/"
        f"licitacion?N={codigo}"
    )


# ============================================================
# EXTRAER DATOS DEL DETALLE
# ============================================================

def extraer_detalle_galicia(codigo):
    """
    Descarga y procesa la página de detalle.
    """

    enlace = construir_enlace_galicia(
        codigo
    )

    try:

        respuesta = session.get(
            enlace,
            timeout=30
        )

        respuesta.raise_for_status()

    except Exception as e:

        print(
            f"Error descargando detalle "
            f"{codigo}: {e}"
        )

        return None

    soup = BeautifulSoup(
        respuesta.content,
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

    # Si no aparece en el DT/DD,
    # buscamos otras estructuras.
    if not titulo:

        titulo_elemento = soup.find(
            string=lambda x:
            x and "Obxecto" in x
        )

        if titulo_elemento:

            padre = titulo_elemento.parent

            if padre:

                siguiente = padre.find_next()

                if siguiente:

                    titulo = normalizar_texto(
                        siguiente.get_text(
                            " ",
                            strip=True
                        )
                    )

    # --------------------------------------------------------
    # TIPO DE CONTRATO
    # --------------------------------------------------------

    tipo_contrato = extraer_dt_dd(
        soup,
        "Tipo de contrato"
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

    # Si no hay presupuesto base,
    # utilizamos el valor estimado como respaldo.
    if importe == 0.0 and valor_estimado > 0:
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

    fecha_publicacion = parsear_fecha_publicacion(
        fecha_publicacion_texto
    )

    # --------------------------------------------------------
    # FECHA LÍMITE
    # --------------------------------------------------------

    fecha_fin_texto = extraer_dt_dd(
        soup,
        "Data e hora límite"
    )

    fecha_fin = parsear_fecha_fin(
        fecha_fin_texto
    )

    if not fecha_fin:
        fecha_fin = "No especificada"

    # --------------------------------------------------------
    # CPV
    # --------------------------------------------------------

    cpv = extraer_cpv(
        soup
    )

    # --------------------------------------------------------
    # NUT / LOCALIZACIÓN
    # --------------------------------------------------------

    lugar = extraer_nuts(
        soup
    )

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

    estado = extraer_estado(
        soup
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    return {
        "enlace": enlace,
        "titulo": titulo,
        "organo": organo,
        "fecha_publicacion": (
            fecha_publicacion.strftime("%Y-%m-%d")
            if fecha_publicacion
            else None
        ),
        "importe": importe,
        "valor_estimado": valor_estimado,
        "tipo_contrato": tipo_contrato,
        "cpv": cpv,
        "lugar": lugar,
        "fecha_fin": fecha_fin,
        "estado": estado
    }


# ============================================================
# OBTENER REGISTROS DEL LISTADO
# ============================================================

def obtener_registros_listado():
    """
    Descarga ultimasPublicaciones.jsp y extrae
    el JSON completo contenido en resSearch.
    """

    print(
        "\nDescargando listado de Galicia..."
    )

    try:

        respuesta = session.get(
            URL_LISTADO_GALICIA,
            timeout=30
        )

        respuesta.raise_for_status()

    except Exception as e:

        print(
            f"Error descargando listado Galicia: {e}"
        )

        return []

    soup = BeautifulSoup(
        respuesta.content,
        "html.parser"
    )

    input_res_search = soup.find(
        "input",
        id="resSearch"
    )

    if not input_res_search:

        print(
            "No se encontró el input resSearch."
        )

        return []

    raw = input_res_search.get(
        "value",
        ""
    )

    if not raw:

        print(
            "resSearch está vacío."
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

    if not isinstance(registros, list):

        print(
            "resSearch no contiene una lista."
        )

        return []

    print(
        f"Registros encontrados en el listado: "
        f"{len(registros)}"
    )

    return registros


# ============================================================
# FILTRO DE ESTADOS
# ============================================================

def estado_no_valido(estado):
    """
    Estados que no queremos incorporar como
    licitaciones activas.
    """

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

def construir_texto_embedding(registro):
    """
    Construye el texto utilizado para generar
    el embedding.
    """

    partes = [
        registro.get("titulo", ""),
        registro.get("organo", ""),
        registro.get("tipo_contrato", ""),
        str(registro.get("importe", "")),
        registro.get("cpv", ""),
        registro.get("lugar", ""),
        registro.get("fecha_fin", "")
    ]

    partes = [
        normalizar_texto(p)
        for p in partes
        if p is not None and str(p).strip()
    ]

    return " | ".join(
        partes
    )


def generar_embedding(registro):
    """
    Genera el embedding del registro.
    """

    texto = construir_texto_embedding(
        registro
    )

    embedding = modelo.encode(
        texto,
        normalize_embeddings=True
    )

    return embedding.tolist()


# ============================================================
# SINCRONIZACIÓN
# ============================================================

def sincronizar_licitaciones_galicia():

    print(
        "\n"
        "============================================================"
    )
    print(
        "SINCRONIZACIÓN LICITACIONES GALICIA"
    )
    print(
        "============================================================"
    )

    hoy = date.today()

    fecha_minima = hoy - timedelta(
        days=DIAS_ATRAS
    )

    print(
        f"Fecha actual: {hoy}"
    )

    print(
        f"Procesando publicaciones desde "
        f"{fecha_minima} hasta {hoy}"
    )

    # ========================================================
    # 1. CARGAR BBDD
    # ========================================================

    print(
        "\nCargando licitaciones existentes..."
    )

    try:

        respuesta_db = (
            supabase
            .table("licitaciones")
            .select("*")
            .execute()
        )

        registros_db = respuesta_db.data or []

    except Exception as e:

        print(
            f"Error cargando BBDD: {e}"
        )

        return

    print(
        f"Licitaciones existentes: "
        f"{len(registros_db)}"
    )

    # ========================================================
    # 2. MAPAS PARA BÚSQUEDA
    # ========================================================

    mapa_enlaces = {}

    registros_existentes = {}

    for registro in registros_db:

        enlace = normalizar_texto(
            registro.get("enlace")
        )

        if enlace:
            mapa_enlaces[enlace] = registro

        titulo = normalizar_texto(
            registro.get("titulo")
        ).lower()

        organo = normalizar_organo(
            registro.get("organo")
        )

        if titulo and organo:

            clave = (
                titulo,
                organo
            )

            registros_existentes[
                clave
            ] = registro

    # ========================================================
    # 3. RESETEAR FLAGS DE GALICIA
    # ========================================================

    print(
        "\nReseteando flags de Galicia..."
    )

    registros_galicia = [
        r
        for r in registros_db
        if contiene_fuente(
            r.get("fuente"),
            FUENTE_GALICIA
        )
    ]

    for i in range(
        0,
        len(registros_galicia),
        50
    ):

        lote = registros_galicia[
            i:i + 50
        ]

        ids = [
            r["id"]
            for r in lote
            if r.get("id") is not None
        ]

        if not ids:
            continue

        try:

            (
                supabase
                .table("licitaciones")
                .update({
                    "es_novedad": False,
                    "es_actualizada": False
                })
                .in_("id", ids)
                .execute()
            )

        except Exception as e:

            print(
                f"Error reseteando flags: {e}"
            )

    # ========================================================
    # 4. OBTENER LISTADO
    # ========================================================

    registros_listado = (
        obtener_registros_listado()
    )

    if not registros_listado:

        print(
            "No se encontraron registros."
        )

        return

    # ========================================================
    # 5. FILTRAR ÚLTIMOS 3 DÍAS
    # ========================================================

    registros_recientes = []

    for item in registros_listado:

        fecha_raw = item.get(
            "fechaPublicacion"
        )

        fecha_publicacion = (
            parsear_fecha_publicacion(
                fecha_raw
            )
        )

        if not fecha_publicacion:
            continue

        if (
            fecha_minima
            <= fecha_publicacion
            <= hoy
        ):

            registros_recientes.append(
                item
            )

    print(
        f"Publicaciones de los últimos 3 días: "
        f"{len(registros_recientes)}"
    )

    # ========================================================
    # CONTADORES
    # ========================================================

    nuevas = 0
    actualizadas = 0
    duplicadas = 0
    sin_cambios = 0
    descartadas_estado = 0
    errores = 0

    lote_nuevos = []

    # Evita duplicados dentro de esta ejecución
    enlaces_procesados = set()
    claves_procesadas = set()

    # ========================================================
    # 6. PROCESAR REGISTROS
    # ========================================================

    for posicion, item in enumerate(
        registros_recientes,
        start=1
    ):

        codigo = item.get(
            "codigo"
        )

        if not codigo:

            continue

        print(
            f"\n[{posicion}/{len(registros_recientes)}] "
            f"Procesando {codigo}..."
        )

        # ----------------------------------------------------
        # EXTRAER DETALLE
        # ----------------------------------------------------

        detalle = extraer_detalle_galicia(
            codigo
        )

        if not detalle:

            errores += 1
            continue

        titulo = normalizar_texto(
            detalle.get("titulo")
        )

        organo = normalizar_texto(
            detalle.get("organo")
        )

        enlace = normalizar_texto(
            detalle.get("enlace")
        )

        if not titulo or not enlace:

            print(
                "  -> Sin título o enlace. Se descarta."
            )

            errores += 1
            continue

        # ----------------------------------------------------
        # ESTADO
        # ----------------------------------------------------

        estado = detalle.get(
            "estado",
            ""
        )

        if estado_no_valido(
            estado
        ):

            print(
                f"  -> Estado excluido: {estado}"
            )

            descartadas_estado += 1
            continue

        # ----------------------------------------------------
        # FECHA PUBLICACIÓN
        # ----------------------------------------------------

        fecha_publicacion = detalle.get(
            "fecha_publicacion"
        )

        if not fecha_publicacion:

            fecha_publicacion_lista = (
                parsear_fecha_publicacion(
                    item.get(
                        "fechaPublicacion"
                    )
                )
            )

            if fecha_publicacion_lista:

                fecha_publicacion = (
                    fecha_publicacion_lista
                    .strftime("%Y-%m-%d")
                )

        # ----------------------------------------------------
        # FILTRO DE FECHA
        # ----------------------------------------------------

        if fecha_publicacion:

            try:

                fecha_pub = datetime.strptime(
                    fecha_publicacion,
                    "%Y-%m-%d"
                ).date()

                if not (
                    fecha_minima
                    <= fecha_pub
                    <= hoy
                ):

                    print(
                        "  -> Fuera de los últimos 3 días."
                    )

                    continue

            except ValueError:

                pass

        # ----------------------------------------------------
        # DUPLICADO DENTRO DE LA SESIÓN
        # ----------------------------------------------------

        if enlace in enlaces_procesados:

            print(
                "  -> Duplicado dentro de esta ejecución."
            )

            duplicadas += 1
            continue

        clave = (
            titulo.lower(),
            normalizar_organo(organo)
        )

        if clave in claves_procesadas:

            print(
                "  -> Duplicado por título + órgano "
                "dentro de esta ejecución."
            )

            duplicadas += 1
            continue

        # ----------------------------------------------------
        # MARCAR COMO PROCESADO
        # ----------------------------------------------------

        enlaces_procesados.add(
            enlace
        )

        # ====================================================
        # 7. BUSCAR POR ENLACE
        # ====================================================

        existente = mapa_enlaces.get(
            enlace
        )

        # ====================================================
        # 8. SI NO EXISTE, BUSCAR POR TÍTULO + ÓRGANO
        # ====================================================

        if not existente:

            existente = registros_existentes.get(
                clave
            )

        # ====================================================
        # 9. REGISTRO EXISTENTE
        # ====================================================

        if existente:

            registro_id = existente.get(
                "id"
            )

            fuente_actual = str(
                existente.get("fuente") or ""
            )

            nueva_fuente = añadir_fuente(
                fuente_actual,
                FUENTE_GALICIA
            )

            cambios = {}

            # ------------------------------------------------
            # COMPROBAR CAMBIOS
            # ------------------------------------------------

            if (
                normalizar_texto(
                    existente.get("titulo")
                )
                != titulo
            ):
                cambios["titulo"] = titulo

            if (
                normalizar_texto(
                    existente.get("organo")
                )
                != organo
            ):
                cambios["organo"] = organo

            if (
                normalizar_texto(
                    existente.get(
                        "fecha_publicacion"
                    )
                )
                != normalizar_texto(
                    fecha_publicacion
                )
            ):
                cambios[
                    "fecha_publicacion"
                ] = fecha_publicacion

            importe_nuevo = detalle.get(
                "importe",
                0.0
            )

            try:
                importe_existente = float(
                    existente.get(
                        "importe"
                    ) or 0.0
                )
            except Exception:
                importe_existente = 0.0

            if abs(
                importe_existente
                - importe_nuevo
            ) > 0.01:

                cambios[
                    "importe"
                ] = importe_nuevo

            tipo_nuevo = normalizar_texto(
                detalle.get(
                    "tipo_contrato"
                )
            )

            tipo_existente = normalizar_texto(
                existente.get(
                    "tipo_contrato"
                )
            )

            if (
                tipo_nuevo
                and tipo_nuevo
                != tipo_existente
            ):

                cambios[
                    "tipo_contrato"
                ] = tipo_nuevo

            cpv_nuevo = normalizar_texto(
                detalle.get(
                    "cpv"
                )
            )

            cpv_existente = normalizar_texto(
                existente.get(
                    "cpv"
                )
            )

            if (
                cpv_nuevo
                and cpv_nuevo
                != cpv_existente
            ):

                cambios[
                    "cpv"
                ] = cpv_nuevo

            lugar_nuevo = normalizar_texto(
                detalle.get(
                    "lugar"
                )
            )

            lugar_existente = normalizar_texto(
                existente.get(
                    "lugar"
                )
            )

            if (
                lugar_nuevo
                and lugar_nuevo
                != lugar_existente
            ):

                cambios[
                    "lugar"
                ] = lugar_nuevo

            fecha_fin_nueva = normalizar_texto(
                detalle.get(
                    "fecha_fin"
                )
            )

            fecha_fin_existente = normalizar_texto(
                existente.get(
                    "fecha_fin"
                )
            )

            if (
                fecha_fin_nueva
                and fecha_fin_nueva
                != fecha_fin_existente
            ):

                cambios[
                    "fecha_fin"
                ] = fecha_fin_nueva

            # ------------------------------------------------
            # AÑADIR FUENTE SI FALTA
            # ------------------------------------------------

            if nueva_fuente != fuente_actual:

                cambios[
                    "fuente"
                ] = nueva_fuente

            # ------------------------------------------------
            # ACTUALIZAR
            # ------------------------------------------------

            if cambios:

                cambios[
                    "es_novedad"
                ] = False

                # Solo consideramos actualización real
                # si cambió información del registro.
                cambios_reales = {
                    k: v
                    for k, v in cambios.items()
                    if k not in {
                        "fuente"
                    }
                }

                cambios[
                    "es_actualizada"
                ] = bool(
                    cambios_reales
                )

                # ------------------------------------------------
                # Si hay cambios de datos, regeneramos embedding
                # ------------------------------------------------

                if cambios_reales:

                    registro_embedding = {
                        **existente,
                        **cambios,
                        "titulo": titulo,
                        "organo": organo,
                        "importe": importe_nuevo,
                        "tipo_contrato": tipo_nuevo,
                        "cpv": cpv_nuevo,
                        "lugar": lugar_nuevo,
                        "fecha_fin": fecha_fin_nueva
                    }

                    try:

                        cambios[
                            "embedding"
                        ] = generar_embedding(
                            registro_embedding
                        )

                    except Exception as e:

                        print(
                            f"  -> Error generando embedding: {e}"
                        )

                try:

                    (
                        supabase
                        .table("licitaciones")
                        .update(cambios)
                        .eq("id", registro_id)
                        .execute()
                    )

                    if cambios_reales:

                        actualizadas += 1

                        print(
                            "  -> Licitación actualizada."
                        )

                    else:

                        print(
                            "  -> Fuente Galicia añadida."
                        )

                    # Actualizar copia local
                    existente.update(
                        cambios
                    )

                    mapa_enlaces[
                        enlace
                    ] = existente

                    registros_existentes[
                        clave
                    ] = existente

                except Exception as e:

                    print(
                        f"  -> Error actualizando: {e}"
                    )

                    errores += 1

            else:

                sin_cambios += 1

                print(
                    "  -> Sin cambios."
                )

            claves_procesadas.add(
                clave
            )

            continue

        # ====================================================
        # 10. REGISTRO NUEVO
        # ====================================================

        print(
            "  -> Nueva licitación."
        )

        try:

            embedding = generar_embedding(
                detalle
            )

        except Exception as e:

            print(
                f"  -> Error generando embedding: {e}"
            )

            errores += 1
            continue

        nuevo_registro = {
            "enlace": enlace,
            "titulo": titulo,
            "organo": organo,
            "fecha_publicacion": fecha_publicacion,
            "importe": detalle.get(
                "importe",
                0.0
            ),
            "tipo_contrato": detalle.get(
                "tipo_contrato",
                ""
            ),
            "cpv": detalle.get(
                "cpv",
                ""
            ),
            "lugar": detalle.get(
                "lugar",
                ""
            ),
            "fecha_fin": detalle.get(
                "fecha_fin",
                "No especificada"
            ),
            "fuente": FUENTE_GALICIA,
            "embedding": embedding,
            "es_novedad": True,
            "es_actualizada": False
        }

        lote_nuevos.append(
            nuevo_registro
        )

        nuevas += 1

        # ----------------------------------------------------
        # Actualizar mapas en memoria
        # ----------------------------------------------------

        mapa_enlaces[
            enlace
        ] = nuevo_registro

        registros_existentes[
            clave
        ] = nuevo_registro

        claves_procesadas.add(
            clave
        )

        # ====================================================
        # INSERTAR POR LOTES
        # ====================================================

        if len(lote_nuevos) >= BATCH_SIZE:

            try:

                (
                    supabase
                    .table("licitaciones")
                    .insert(lote_nuevos)
                    .execute()
                )

                print(
                    f"  -> Insertado lote de "
                    f"{len(lote_nuevos)} nuevas."
                )

                lote_nuevos = []

            except Exception as e:

                print(
                    f"Error insertando lote: {e}"
                )

                errores += len(
                    lote_nuevos
                )

                lote_nuevos = []

        # Pequeña pausa para no saturar Galicia
        time.sleep(0.15)

    # ========================================================
    # 11. INSERTAR ÚLTIMO LOTE
    # ========================================================

    if lote_nuevos:

        try:

            (
                supabase
                .table("licitaciones")
                .insert(lote_nuevos)
                .execute()
            )

            print(
                f"\nInsertado último lote de "
                f"{len(lote_nuevos)} nuevas."
            )

        except Exception as e:

            print(
                f"\nError insertando último lote: {e}"
            )

            errores += len(
                lote_nuevos
            )

    # ========================================================
    # 12. LIMPIEZA DE CADUCADAS DE GALICIA
    # ========================================================

    print(
        "\nComprobando licitaciones caducadas de Galicia..."
    )

    ids_a_borrar = []
    ids_a_actualizar = []

    try:

        respuesta_caducadas = (
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

        for item in (
            respuesta_caducadas.data or []
        ):

            fecha_fin = item.get(
                "fecha_fin"
            )

            if (
                not fecha_fin
                or fecha_fin == "No especificada"
            ):
                continue

            try:

                fecha_cierre = datetime.strptime(
                    str(fecha_fin),
                    "%Y-%m-%d"
                ).date()

            except ValueError:

                continue

            if fecha_cierre >= hoy:
                continue

            fuente_actual = str(
                item.get("fuente") or ""
            )

            fuentes = normalizar_fuentes(
                fuente_actual
            )

            # ------------------------------------------------
            # Galicia es la única fuente
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Tiene otras fuentes
            # ------------------------------------------------

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
        # ELIMINAR
        # ----------------------------------------------------

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
                .in_("id", lote_ids)
                .execute()
            )

        if ids_a_borrar:

            print(
                f"Eliminadas {len(ids_a_borrar)} "
                f"licitaciones caducadas cuya única "
                f"fuente era Galicia."
            )

        # ----------------------------------------------------
        # QUITAR SOLO GALICIA
        # ----------------------------------------------------

        for registro_id, nueva_fuente in (
            ids_a_actualizar
        ):

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
                    f"Error quitando Galicia "
                    f"del registro {registro_id}: "
                    f"{e}"
                )

        if ids_a_actualizar:

            print(
                f"Quitada la fuente Galicia de "
                f"{len(ids_a_actualizar)} licitaciones "
                f"caducadas que tenían otras fuentes."
            )

    except Exception as e:

        print(
            "Error en la limpieza de caducadas: "
            f"{e}"
        )

    # ========================================================
    # 13. RESUMEN
    # ========================================================

    print(
        "\n"
        "============================================================"
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
        f"Sin cambios:            {sin_cambios}"
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
