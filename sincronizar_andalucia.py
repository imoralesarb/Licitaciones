# -*- coding: utf-8 -*-

# Editado 10/09/2026
# Sincroniza las licitaciones de la Junta de Andalucía con Supabase.
#
# - Consulta directamente el Elasticsearch utilizado por el portal oficial.
# - Filtra directamente en Elasticsearch las licitaciones publicadas
#   en los últimos 3 días.
# - Recorre únicamente las páginas necesarias dentro de ese periodo.
# - Ordena por fecha de publicación descendente.
# - Consulta el detalle de cada expediente.
# - Busca primero coincidencias por enlace.
# - Si no existe el enlace, busca por título + órgano.
# - Si una licitación ya existe con otra fuente, añade Andalucía a "fuente".
# - Evita duplicados.
# - Marca las nuevas como es_novedad=True.
# - Marca como es_actualizada=True únicamente cuando cambian datos reales.
# - No considera como actualización el simple hecho de añadir una fuente.
# - Genera embedding para las nuevas licitaciones.


from datetime import datetime, date, timedelta
import os
import re
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Faltan las variables de entorno SUPABASE_URL y/o SUPABASE_KEY."
    )

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ------------------------------------------------------------
# URLs de Elasticsearch
# ------------------------------------------------------------

BASE_URL = (
    "https://www.juntadeandalucia.es/"
    "haciendayadministracionpublica/apl/pdc-front-publico"
)

URL_BUSCADOR = (
    f"{BASE_URL}/elastic/"
    "sirec_pdc_expedientes/_search?pretty"
)

URL_DETALLE = (
    f"{BASE_URL}/elastic/"
    "sirec_pdc_expedientes_details/_search?pretty"
)


# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------

FUENTE = "Andalucía"

TAMANO_PAGINA = 10

# Hoy + los dos días anteriores.
# Ejemplo si hoy es 10/09:
# 10/09, 09/09 y 08/09.
DIAS_ATRAS = 2

PAUSA_ENTRE_PETICIONES = 0.15

TAMANO_LOTE = 500


# ------------------------------------------------------------
# Modelo de embeddings
# ------------------------------------------------------------

MODELO_EMBEDDING = "intfloat/multilingual-e5-small"

print("Cargando modelo de embeddings...")

model = SentenceTransformer(
    MODELO_EMBEDDING,
    device="cpu"
)

print("Modelo de embeddings cargado.")


# ============================================================
# 2. SESIÓN HTTP
# ============================================================

session = requests.Session()

retry_strategy = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[
        429,
        500,
        502,
        503,
        504
    ],
    allowed_methods=[
        "GET",
        "POST"
    ],
    raise_on_status=False
)

adapter = HTTPAdapter(
    max_retries=retry_strategy
)

session.mount(
    "http://",
    adapter
)

session.mount(
    "https://",
    adapter
)

session.headers.update({
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0.0.0 Safari/537.36"
    ),
})


# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def limpiar_texto(valor):
    """
    Limpia espacios y devuelve una cadena.
    """

    if valor is None:
        return ""

    texto = str(valor)

    texto = texto.replace("\xa0", " ")
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def normalizar_texto(valor):
    """
    Normaliza texto para comparaciones.
    """

    texto = limpiar_texto(valor).lower()

    texto = (
        texto
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )

    return texto


def normalizar_organo(organo):
    """
    Normaliza el nombre del órgano.
    """

    return normalizar_texto(organo)


def obtener_valor(data, *claves):
    """
    Busca el primer valor disponible entre varias claves.
    """

    if not isinstance(data, dict):
        return None

    for clave in claves:

        if clave in data:

            valor = data[clave]

            if valor is not None and valor != "":
                return valor

    return None


def convertir_importe(valor):
    """
    Convierte importes españoles a float.

    Ejemplo:
        6.021.000,00 € -> 6021000.00
    """

    if valor is None:
        return None

    if isinstance(valor, (int, float)):
        return float(valor)

    texto = limpiar_texto(valor)

    if not texto:
        return None

    texto = texto.replace("€", "")
    texto = texto.replace("EUR", "")
    texto = texto.strip()

    if "," in texto:

        texto = texto.replace(".", "")
        texto = texto.replace(",", ".")

    else:

        texto = texto.replace(" ", "")

    texto = re.sub(
        r"[^\d.\-]",
        "",
        texto
    )

    if not texto:
        return None

    try:
        return float(texto)

    except ValueError:
        return None


def extraer_fecha(valor):
    """
    Convierte diferentes formatos de fecha a datetime.
    """

    if valor is None:
        return None

    if isinstance(valor, datetime):
        return valor

    if isinstance(valor, date):
        return datetime.combine(
            valor,
            datetime.min.time()
        )

    texto = limpiar_texto(valor)

    if not texto:
        return None

    formatos = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y"
    ]

    for formato in formatos:

        try:
            return datetime.strptime(
                texto,
                formato
            )

        except ValueError:
            continue

    try:

        texto_iso = texto.replace(
            "Z",
            "+00:00"
        )

        return datetime.fromisoformat(
            texto_iso
        ).replace(
            tzinfo=None
        )

    except ValueError:
        pass

    return None


def formatear_fecha_supabase(fecha):
    """
    Convierte datetime a formato ISO para Supabase.
    """

    if fecha is None:
        return None

    return fecha.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def obtener_fuentes(fuente):
    """
    Convierte el campo fuente en una lista.

    Ejemplo:
        Galicia, TED
        ->
        ["Galicia", "TED"]
    """

    if not fuente:
        return []

    return [
        parte.strip()
        for parte in str(fuente).split(",")
        if parte.strip()
    ]


def contiene_fuente(fuente, fuente_buscar):
    """
    Comprueba si una fuente está presente.
    """

    fuentes = obtener_fuentes(
        fuente
    )

    return any(
        normalizar_texto(f) ==
        normalizar_texto(fuente_buscar)
        for f in fuentes
    )


def añadir_fuente(
    fuente_actual,
    nueva_fuente
):
    """
    Añade una fuente sin duplicarla.
    """

    fuentes = obtener_fuentes(
        fuente_actual
    )

    if not any(
        normalizar_texto(f) ==
        normalizar_texto(nueva_fuente)
        for f in fuentes
    ):
        fuentes.append(
            nueva_fuente
        )

    return ", ".join(fuentes)


def eliminar_fuente(
    fuente_actual,
    fuente_eliminar
):
    """
    Elimina una fuente concreta.
    """

    fuentes = obtener_fuentes(
        fuente_actual
    )

    fuentes = [
        f
        for f in fuentes
        if normalizar_texto(f) !=
        normalizar_texto(fuente_eliminar)
    ]

    return ", ".join(fuentes)


def extraer_codigo_expediente(url):
    """
    Extrae idExpediente de una URL de Andalucía.
    """

    if not url:
        return None

    coincidencia = re.search(
        r"idExpediente=([^&#]+)",
        str(url),
        flags=re.I
    )

    if coincidencia:
        return coincidencia.group(1)

    return None


def construir_enlace(codigo):
    """
    Construye la URL pública de detalle.
    """

    if not codigo:
        return None

    return (
        f"{BASE_URL}/perfiles-licitaciones/"
        f"detalle-licitacion?idExpediente={codigo}"
    )


def normalizar_tipo_contrato(tipo):
    """
    Normaliza los tipos de contrato.
    """

    tipo_limpio = limpiar_texto(
        tipo
    )

    if not tipo_limpio:
        return None

    tipo_norm = normalizar_texto(
        tipo_limpio
    )

    if tipo_norm.startswith(
        "suministr"
    ):
        return "Suministro"

    if tipo_norm.startswith(
        "servici"
    ):
        return "Servicios"

    if tipo_norm == "obras":
        return "Obras"

    if (
        "concesion de servicios"
        in tipo_norm
    ):
        return "Concesión de servicios"

    if (
        "concesion de obras"
        in tipo_norm
    ):
        return "Concesión de obras"

    return tipo_limpio


def limpiar_lugar_ejecucion(lugar):
    """
    Normaliza el lugar de ejecución.

    Ejemplo:
        ES618 - Sevilla
        ->
        Sevilla
    """

    if lugar is None:
        return None

    if isinstance(lugar, list):

        lugares = []

        for elemento in lugar:

            limpio = limpiar_lugar_ejecucion(
                elemento
            )

            if limpio:
                lugares.append(
                    limpio
                )

        if not lugares:
            return None

        resultado = []

        for item in lugares:

            if item not in resultado:
                resultado.append(
                    item
                )

        return ", ".join(
            resultado
        )

    texto = limpiar_texto(
        lugar
    )

    if not texto:
        return None

    texto = re.sub(
        r"\(c[oó]digo\s+NUTS\)",
        "",
        texto,
        flags=re.I
    )

    texto = re.sub(
        r"\bES\d{3}\s*[-–—:]\s*",
        "",
        texto,
        flags=re.I
    )

    texto = re.sub(
        r"\bES\d{3}\b",
        "",
        texto,
        flags=re.I
    )

    texto = limpiar_texto(
        texto
    )

    return texto or None


def extraer_codigo_cpv(cpv):
    """
    Extrae el CPV.
    """

    if cpv is None:
        return None

    if isinstance(cpv, list):

        valores = []

        for elemento in cpv:

            valor = extraer_codigo_cpv(
                elemento
            )

            if valor:
                valores.append(
                    valor
                )

        if not valores:
            return None

        resultado = []

        for valor in valores:

            if valor not in resultado:
                resultado.append(
                    valor
                )

        return ", ".join(
            resultado
        )

    if isinstance(cpv, dict):

        codigo = obtener_valor(
            cpv,
            "codigo",
            "code",
            "cpv",
            "codigoCPV"
        )

        descripcion = obtener_valor(
            cpv,
            "descripcion",
            "description",
            "nombre",
            "name"
        )

        if codigo and descripcion:

            return (
                f"{limpiar_texto(codigo)} "
                f"{limpiar_texto(descripcion)}"
            )

        if codigo:
            return limpiar_texto(
                codigo
            )

        if descripcion:
            return limpiar_texto(
                descripcion
            )

        return None

    texto = limpiar_texto(
        cpv
    )

    return texto or None


def extraer_texto_recursivo(obj):
    """
    Convierte una estructura JSON en texto.
    """

    partes = []

    if obj is None:
        return partes

    if isinstance(obj, dict):

        for clave, valor in obj.items():

            if clave in {
                "embedding",
                "vector",
                "_source",
                "_index",
                "_id",
                "sort"
            }:
                continue

            if isinstance(
                valor,
                (dict, list)
            ):

                partes.extend(
                    extraer_texto_recursivo(
                        valor
                    )
                )

            else:

                texto = limpiar_texto(
                    valor
                )

                if texto:

                    partes.append(
                        f"{limpiar_texto(clave)}: "
                        f"{texto}"
                    )

    elif isinstance(obj, list):

        for elemento in obj:

            partes.extend(
                extraer_texto_recursivo(
                    elemento
                )
            )

    else:

        texto = limpiar_texto(
            obj
        )

        if texto:
            partes.append(
                texto
            )

    return partes


# ============================================================
# 4. PETICIONES A ELASTICSEARCH
# ============================================================

def consultar_elasticsearch(
    url,
    payload,
    timeout=60
):
    """
    Ejecuta una consulta POST contra Elasticsearch.
    """

    try:

        respuesta = session.post(
            url,
            json=payload,
            timeout=timeout
        )

        respuesta.raise_for_status()

        return respuesta.json()

    except requests.RequestException as e:

        print(
            f"ERROR HTTP Elasticsearch: {e}"
        )

        return None

    except ValueError as e:

        print(
            "ERROR al interpretar JSON "
            f"de Elasticsearch: {e}"
        )

        return None


def obtener_pagina_busqueda(
    desde,
    fecha_minima,
    fecha_maxima
):
    """
    Obtiene una página de licitaciones publicadas
    dentro del periodo solicitado.

    IMPORTANTE:
    El filtro de fecha se hace directamente en Elasticsearch,
    por lo que no se recorren las 87.000+ licitaciones históricas.
    """

    payload = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "fechaPublicacion": {
                                "gte": (
                                    fecha_minima.strftime(
                                        "%Y-%m-%dT00:00:00"
                                    )
                                ),
                                "lt": (
                                    fecha_maxima.strftime(
                                        "%Y-%m-%dT00:00:00"
                                    )
                                )
                            }
                        }
                    }
                ],
                "must_not": [
                    {
                        "match": {
                            "estado.codigo": {
                                "query": "BRR"
                            }
                        }
                    },
                    {
                        "match": {
                            "codigoProcedimiento": 9
                        }
                    }
                ],
                "should": []
            }
        },
        "size": TAMANO_PAGINA,
        "sort": [
            {
                "fechaPublicacion": "desc"
            }
        ],
        "track_total_hits": True,
        "from": desde
    }

    return consultar_elasticsearch(
        URL_BUSCADOR,
        payload
    )


def obtener_detalle(
    codigo_expediente
):
    """
    Obtiene el detalle de un expediente concreto.
    """

    payload = {
        "query": {
            "match": {
                "_id": str(
                    codigo_expediente
                )
            }
        }
    }

    return consultar_elasticsearch(
        URL_DETALLE,
        payload
    )


# ============================================================
# 5. PROCESAMIENTO DE ELASTICSEARCH
# ============================================================

def extraer_hits_busqueda(
    respuesta
):
    """
    Extrae los hits de una respuesta.
    """

    if not respuesta:
        return []

    hits = respuesta.get(
        "hits",
        {}
    )

    if not isinstance(
        hits,
        dict
    ):
        return []

    resultados = hits.get(
        "hits",
        []
    )

    if not isinstance(
        resultados,
        list
    ):
        return []

    return resultados


def obtener_total_busqueda(
    respuesta
):
    """
    Obtiene el número total de resultados.
    """

    if not respuesta:
        return 0

    hits = respuesta.get(
        "hits",
        {}
    )

    if not isinstance(
        hits,
        dict
    ):
        return 0

    total = hits.get(
        "total",
        0
    )

    if isinstance(
        total,
        dict
    ):

        return int(
            total.get(
                "value",
                0
            )
        )

    try:
        return int(
            total
        )

    except (
        TypeError,
        ValueError
    ):
        return 0


def obtener_source(hit):
    """
    Obtiene _source.
    """

    if not isinstance(
        hit,
        dict
    ):
        return {}

    source = hit.get(
        "_source",
        {}
    )

    if isinstance(
        source,
        dict
    ):
        return source

    return {}


def extraer_fecha_publicacion_hit(
    hit
):
    """
    Obtiene fechaPublicacion.
    """

    source = obtener_source(
        hit
    )

    valor = obtener_valor(
        source,
        "fechaPublicacion",
        "fecha_publicacion",
        "fecha"
    )

    if (
        valor is None
        and isinstance(hit, dict)
    ):

        sort_values = hit.get(
            "sort"
        )

        if (
            isinstance(
                sort_values,
                list
            )
            and sort_values
        ):
            valor = sort_values[0]

    return extraer_fecha(
        valor
    )


def extraer_codigo_hit(
    hit
):
    """
    Obtiene el identificador del expediente.
    """

    if not isinstance(
        hit,
        dict
    ):
        return None

    codigo = hit.get(
        "_id"
    )

    if codigo:
        return str(
            codigo
        )

    source = obtener_source(
        hit
    )

    codigo = obtener_valor(
        source,
        "idExpediente",
        "id",
        "codigoExpediente",
        "numeroExpediente"
    )

    if codigo:
        return str(
            codigo
        )

    return None


# ============================================================
# 6. OBTENER LICITACIONES DE LOS ÚLTIMOS 3 DÍAS
# ============================================================

def obtener_licitaciones_ultimos_tres_dias():
    """
    Obtiene únicamente las licitaciones publicadas
    en los últimos 3 días.

    El filtro se ejecuta directamente en Elasticsearch.
    """

    hoy = date.today()

    fecha_minima = hoy - timedelta(
        days=DIAS_ATRAS
    )

    # Se utiliza mañana como límite exclusivo.
    #
    # Si hoy es 10/09/2026:
    #
    # gte -> 08/09/2026 00:00
    # lt  -> 11/09/2026 00:00
    #
    # Esto incluye todo el día 10/09.

    fecha_maxima = hoy + timedelta(
        days=1
    )

    print()
    print("=" * 70)
    print("BUSCADOR DE ANDALUCÍA")
    print("=" * 70)

    print(
        f"Fecha actual: "
        f"{hoy.strftime('%d/%m/%Y')}"
    )

    print(
        "Periodo de publicación: "
        f"{fecha_minima.strftime('%d/%m/%Y')} - "
        f"{hoy.strftime('%d/%m/%Y')}"
    )

    print("=" * 70)

    desde = 0

    resultados_periodo = []

    total_periodo = None

    while True:

        print(
            f"\nConsultando resultados "
            f"{desde + 1}-"
            f"{desde + TAMANO_PAGINA} "
            "del periodo..."
        )

        respuesta = obtener_pagina_busqueda(
            desde,
            fecha_minima,
            fecha_maxima
        )

        if respuesta is None:

            print(
                "No se pudo obtener la página."
            )

            break

        hits = extraer_hits_busqueda(
            respuesta
        )

        total_periodo = obtener_total_busqueda(
            respuesta
        )

        if desde == 0:

            print(
                "Total de licitaciones publicadas "
                "en el periodo: "
                f"{total_periodo}"
            )

        if not hits:
            break

        for hit in hits:

            fecha_publicacion = (
                extraer_fecha_publicacion_hit(
                    hit
                )
            )

            codigo = extraer_codigo_hit(
                hit
            )

            if not codigo:
                continue

            if fecha_publicacion is None:
                continue

            resultados_periodo.append({
                "hit": hit,
                "codigo": codigo,
                "fecha_publicacion": (
                    fecha_publicacion
                )
            })

        if (
            desde + TAMANO_PAGINA
            >= total_periodo
        ):
            break

        if len(hits) < TAMANO_PAGINA:
            break

        desde += TAMANO_PAGINA

        time.sleep(
            PAUSA_ENTRE_PETICIONES
        )

    print()

    print(
        "Total de licitaciones recuperadas "
        "del periodo: "
        f"{len(resultados_periodo)}"
    )

    return resultados_periodo


# ============================================================
# 7. CARGAR LICITACIONES EXISTENTES DE SUPABASE
# ============================================================

def cargar_licitaciones_existentes():
    """
    Carga las licitaciones existentes de Supabase.

    Se crean dos índices:
        - por enlace
        - por título + órgano
    """

    print()
    print(
        "Cargando licitaciones existentes "
        "de Supabase..."
    )

    registros = []

    inicio = 0

    while True:

        fin = (
            inicio
            + TAMANO_LOTE
            - 1
        )

        respuesta = (
            supabase
            .table("licitaciones")
            .select(
                "id, enlace, titulo, organo, fuente, "
                "fecha, importe, tipo_contrato, cpv, "
                "fecha_fin, lugar_ejecucion, "
                "texto_completo, embedding, "
                "es_novedad, es_actualizada"
            )
            .range(
                inicio,
                fin
            )
            .execute()
        )

        lote = respuesta.data or []

        if not lote:
            break

        registros.extend(
            lote
        )

        if len(lote) < TAMANO_LOTE:
            break

        inicio += TAMANO_LOTE

    print(
        "Licitaciones cargadas de Supabase: "
        f"{len(registros)}"
    )

    por_enlace = {}

    por_titulo_organo = {}

    for registro in registros:

        enlace = limpiar_texto(
            registro.get(
                "enlace"
            )
        )

        if enlace:
            por_enlace[enlace] = registro

        titulo = limpiar_texto(
            registro.get(
                "titulo"
            )
        )

        organo = normalizar_organo(
            registro.get(
                "organo"
            )
        )

        if titulo and organo:

            clave = (
                normalizar_texto(
                    titulo
                ),
                organo
            )

            if clave not in por_titulo_organo:

                por_titulo_organo[
                    clave
                ] = registro

    return (
        registros,
        por_enlace,
        por_titulo_organo
    )


# ============================================================
# 8. GENERAR EMBEDDING
# ============================================================

def generar_embedding(
    registro
):
    """
    Genera el embedding de una licitación.
    """

    partes = []

    if registro.get(
        "titulo"
    ):
        partes.append(
            f"Título: "
            f"{registro['titulo']}"
        )

    if registro.get(
        "organo"
    ):
        partes.append(
            f"Órgano: "
            f"{registro['organo']}"
        )

    if registro.get(
        "tipo_contrato"
    ):
        partes.append(
            "Tipo de contrato: "
            f"{registro['tipo_contrato']}"
        )

    if registro.get(
        "cpv"
    ):
        partes.append(
            f"CPV: "
            f"{registro['cpv']}"
        )

    if registro.get(
        "lugar_ejecucion"
    ):
        partes.append(
            "Lugar: "
            f"{registro['lugar_ejecucion']}"
        )

    if registro.get(
        "texto_completo"
    ):
        partes.append(
            registro[
                "texto_completo"
            ]
        )

    texto = " | ".join(
        partes
    )

    if not texto.strip():
        return None

    try:

        embedding = model.encode(
            texto,
            normalize_embeddings=True
        )

        return embedding.tolist()

    except Exception as e:

        print(
            f"ERROR generando embedding: {e}"
        )

        return None


# ============================================================
# 9. COMPARAR CAMBIOS
# ============================================================

def valores_diferentes(
    valor_nuevo,
    valor_antiguo
):
    """
    Compara valores de forma robusta.
    """

    if (
        valor_nuevo is None
        and valor_antiguo in (
            None,
            ""
        )
    ):
        return False

    if (
        valor_antiguo is None
        and valor_nuevo in (
            None,
            ""
        )
    ):
        return False

    if (
        isinstance(
            valor_nuevo,
            float
        )
        and isinstance(
            valor_antiguo,
            (float, int)
        )
    ):

        return abs(
            valor_nuevo
            - float(valor_antiguo)
        ) > 0.000001

    return (
        str(valor_nuevo).strip()
        !=
        str(valor_antiguo).strip()
    )


def hay_cambios_reales(
    nuevo,
    existente
):
    """
    Determina si una licitación ha cambiado realmente.

    El campo fuente no se considera cambio real.
    """

    campos = [
        "titulo",
        "organo",
        "importe",
        "tipo_contrato",
        "cpv",
        "fecha_fin",
        "lugar_ejecucion",
        "texto_completo"
    ]

    for campo in campos:

        valor_nuevo = nuevo.get(
            campo
        )

        valor_antiguo = existente.get(
            campo
        )

        if campo == "fecha_fin":

            if isinstance(
                valor_nuevo,
                datetime
            ):
                valor_nuevo = (
                    formatear_fecha_supabase(
                        valor_nuevo
                    )
                )

            if valor_antiguo:

                fecha_antigua = extraer_fecha(
                    valor_antiguo
                )

                if fecha_antigua:

                    valor_antiguo = (
                        formatear_fecha_supabase(
                            fecha_antigua
                        )
                    )

        if valores_diferentes(
            valor_nuevo,
            valor_antiguo
        ):
            return True

    return False


# ============================================================
# 10. PREPARAR REGISTRO PARA SUPABASE
# ============================================================

def preparar_registro_supabase(
    datos,
    fecha_publicacion,
    embedding=None
):
    """
    Prepara un registro para insertar en Supabase.
    """

    registro = {
        "enlace": datos.get(
            "enlace"
        ),
        "titulo": datos.get(
            "titulo"
        ),
        "organo": datos.get(
            "organo"
        ),
        "fuente": FUENTE,
        "fecha": (
            formatear_fecha_supabase(
                fecha_publicacion
            )
        ),
        "importe": datos.get(
            "importe"
        ),
        "tipo_contrato": datos.get(
            "tipo_contrato"
        ),
        "cpv": datos.get(
            "cpv"
        ),
        "fecha_fin": (
            formatear_fecha_supabase(
                datos.get(
                    "fecha_fin"
                )
            )
        ),
        "lugar_ejecucion": datos.get(
            "lugar_ejecucion"
        ),
        "texto_completo": datos.get(
            "texto_completo"
        ),
        "es_novedad": True,
        "es_actualizada": False
    }

    if embedding is not None:

        registro[
            "embedding"
        ] = embedding

    return registro


# ============================================================
# 11. PROCESAR LICITACIÓN
# ============================================================

def procesar_licitacion(
    datos,
    fecha_publicacion,
    por_enlace,
    por_titulo_organo,
    contadores
):
    """
    Procesa una licitación individual.
    """

    enlace = limpiar_texto(
        datos.get(
            "enlace"
        )
    )

    titulo = limpiar_texto(
        datos.get(
            "titulo"
        )
    )

    organo = limpiar_texto(
        datos.get(
            "organo"
        )
    )

    if not enlace or not titulo:

        contadores[
            "errores"
        ] += 1

        print(
            "  [ERROR] Licitación sin "
            "enlace o título."
        )

        return

    # --------------------------------------------------------
    # Buscar por enlace
    # --------------------------------------------------------

    existente = por_enlace.get(
        enlace
    )

    # --------------------------------------------------------
    # Buscar por título + órgano
    # --------------------------------------------------------

    if (
        existente is None
        and organo
    ):

        clave = (
            normalizar_texto(
                titulo
            ),
            normalizar_organo(
                organo
            )
        )

        existente = (
            por_titulo_organo.get(
                clave
            )
        )

        if existente is not None:

            contadores[
                "duplicadas_titulo_organo"
            ] += 1

            print(
                "  [DUPLICADA] Coincidencia "
                "por título + órgano."
            )

    # --------------------------------------------------------
    # NUEVA
    # --------------------------------------------------------

    if existente is None:

        print(
            f"  [NUEVA] {titulo}"
        )

        embedding = generar_embedding(
            datos
        )

        registro = (
            preparar_registro_supabase(
                datos,
                fecha_publicacion,
                embedding
            )
        )

        try:

            respuesta = (
                supabase
                .table("licitaciones")
                .insert(
                    registro
                )
                .execute()
            )

            insertados = (
                respuesta.data or []
            )

            if not insertados:

                raise RuntimeError(
                    "Supabase no devolvió "
                    "el registro insertado."
                )

            nuevo_registro = (
                insertados[0]
            )

            nuevo_registro.setdefault(
                "fuente",
                FUENTE
            )

            por_enlace[
                enlace
            ] = nuevo_registro

            clave = (
                normalizar_texto(
                    titulo
                ),
                normalizar_organo(
                    organo
                )
            )

            if clave[0] and clave[1]:

                por_titulo_organo[
                    clave
                ] = nuevo_registro

            contadores[
                "nuevas"
            ] += 1

        except Exception as e:

            contadores[
                "errores"
            ] += 1

            print(
                f"  [ERROR INSERTANDO] {e}"
            )

        return

    # --------------------------------------------------------
    # YA EXISTE
    # --------------------------------------------------------

    fuente_actual = existente.get(
        "fuente"
    )

    tiene_andalucia = contiene_fuente(
        fuente_actual,
        FUENTE
    )

    cambios = hay_cambios_reales(
        datos,
        existente
    )

    actualizacion = {}

    campos = [
        "titulo",
        "organo",
        "importe",
        "tipo_contrato",
        "cpv",
        "lugar_ejecucion",
        "texto_completo"
    ]

    for campo in campos:

        nuevo = datos.get(
            campo
        )

        if nuevo is not None:

            actualizacion[
                campo
            ] = nuevo

    if datos.get(
        "fecha_fin"
    ) is not None:

        actualizacion[
            "fecha_fin"
        ] = (
            formatear_fecha_supabase(
                datos[
                    "fecha_fin"
                ]
            )
        )

    if fecha_publicacion is not None:

        actualizacion[
            "fecha"
        ] = (
            formatear_fecha_supabase(
                fecha_publicacion
            )
        )

    # --------------------------------------------------------
    # Añadir Andalucía como fuente
    # --------------------------------------------------------

    if not tiene_andalucia:

        actualizacion[
            "fuente"
        ] = añadir_fuente(
            fuente_actual,
            FUENTE
        )

    # --------------------------------------------------------
    # Marcar actualización
    # --------------------------------------------------------

    if cambios:

        actualizacion[
            "es_actualizada"
        ] = True

    # Si únicamente se ha añadido Andalucía
    # como fuente, NO se marca como actualizada.

    # --------------------------------------------------------
    # Actualizar Supabase
    # --------------------------------------------------------

    if not actualizacion:

        contadores[
            "existentes"
        ] += 1

        print(
            "  [EXISTENTE] Sin cambios."
        )

        return

    try:

        (
            supabase
            .table("licitaciones")
            .update(
                actualizacion
            )
            .eq(
                "id",
                existente["id"]
            )
            .execute()
        )

        if cambios:

            contadores[
                "actualizadas"
            ] += 1

            print(
                "  [ACTUALIZADA] "
                f"{titulo}"
            )

        else:

            contadores[
                "existentes"
            ] += 1

            if not tiene_andalucia:

                print(
                    "  [FUENTE AÑADIDA] "
                    f"{titulo}"
                )

            else:

                print(
                    "  [EXISTENTE] "
                    "Sin cambios."
                )

        existente.update(
            actualizacion
        )

        por_enlace[
            enlace
        ] = existente

        clave = (
            normalizar_texto(
                titulo
            ),
            normalizar_organo(
                organo
            )
        )

        if clave[0] and clave[1]:

            por_titulo_organo[
                clave
            ] = existente

    except Exception as e:

        contadores[
            "errores"
        ] += 1

        print(
            f"  [ERROR ACTUALIZANDO] {e}"
        )


# ============================================================
# 12. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    inicio_total = time.time()

    print()
    print("=" * 70)
    print(
        "SINCRONIZACIÓN DE LICITACIONES - ANDALUCÍA"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Obtener licitaciones de los últimos 3 días
    # --------------------------------------------------------

    resultados = (
        obtener_licitaciones_ultimos_tres_dias()
    )

    if not resultados:

        print()
        print(
            "No se han encontrado licitaciones "
            "dentro del periodo."
        )

        print()
        print("=" * 70)
        print(
            "SINCRONIZACIÓN FINALIZADA"
        )
        print("=" * 70)

        return

    # --------------------------------------------------------
    # Cargar Supabase
    # --------------------------------------------------------

    (
        registros_existentes,
        por_enlace,
        por_titulo_organo
    ) = cargar_licitaciones_existentes()

    # --------------------------------------------------------
    # Contadores
    # --------------------------------------------------------

    contadores = {
        "nuevas": 0,
        "actualizadas": 0,
        "existentes": 0,
        "duplicadas_titulo_organo": 0,
        "errores": 0
    }

    # --------------------------------------------------------
    # Procesar licitaciones
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"PROCESANDO {len(resultados)} LICITACIONES"
    )
    print("=" * 70)

    procesadas = set()

    for numero, resultado in enumerate(
        resultados,
        start=1
    ):

        codigo = resultado.get(
            "codigo"
        )

        fecha_publicacion = (
            resultado.get(
                "fecha_publicacion"
            )
        )

        if not codigo:

            contadores[
                "errores"
            ] += 1

            print(
                f"\n[{numero}/{len(resultados)}] "
                "[ERROR] Sin identificador."
            )

            continue

        codigo = str(
            codigo
        )

        # Evitar duplicados dentro de la propia consulta.
        if codigo in procesadas:

            print(
                f"\n[{numero}/{len(resultados)}] "
                f"[DUPLICADO] Expediente {codigo}"
            )

            continue

        procesadas.add(
            codigo
        )

        print()
        print(
            f"[{numero}/{len(resultados)}] "
            f"Expediente: {codigo}"
        )

        # ----------------------------------------------------
        # Obtener detalle
        # ----------------------------------------------------

        respuesta_detalle = (
            obtener_detalle(
                codigo
            )
        )

        if respuesta_detalle is None:

            contadores[
                "errores"
            ] += 1

            print(
                "  [ERROR] No se pudo obtener "
                "el detalle."
            )

            continue

        datos = (
            extraer_datos_detalle(
                respuesta_detalle,
                codigo
            )
        )

        if not datos:

            contadores[
                "errores"
            ] += 1

            print(
                "  [ERROR] No se pudieron "
                "extraer los datos."
            )

            continue

        print(
            "  Título: "
            f"{datos.get('titulo') or 'SIN TÍTULO'}"
        )

        print(
            "  Órgano: "
            f"{datos.get('organo') or 'SIN ÓRGANO'}"
        )

        print(
            "  Tipo: "
            f"{datos.get('tipo_contrato') or 'SIN TIPO'}"
        )

        print(
            "  Importe: "
            f"{datos.get('importe')}"
        )

        print(
            "  Fecha fin: "
            f"{formatear_fecha_supabase(datos.get('fecha_fin'))}"
        )

        # ----------------------------------------------------
        # Sincronizar con Supabase
        # ----------------------------------------------------

        procesar_licitacion(
            datos,
            fecha_publicacion,
            por_enlace,
            por_titulo_organo,
            contadores
        )

        time.sleep(
            PAUSA_ENTRE_PETICIONES
        )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    tiempo_total = (
        time.time()
        - inicio_total
    )

    print()
    print("=" * 70)
    print(
        "RESUMEN DE SINCRONIZACIÓN"
    )
    print("=" * 70)

    print(
        "Registros publicados en los últimos "
        "3 días: "
        f"{len(resultados)}"
    )

    print(
        "Nuevas insertadas: "
        f"{contadores['nuevas']}"
    )

    print(
        "Actualizadas: "
        f"{contadores['actualizadas']}"
    )

    print(
        "Ya existentes sin cambios: "
        f"{contadores['existentes']}"
    )

    print(
        "Duplicadas por título + órgano: "
        f"{contadores['duplicadas_titulo_organo']}"
    )

    print(
        "Errores: "
        f"{contadores['errores']}"
    )

    print(
        f"Tiempo total: "
        f"{tiempo_total:.2f} segundos"
    )

    print("=" * 70)
    print(
        "SINCRONIZACIÓN FINALIZADA"
    )
    print("=" * 70)


# ============================================================
# 13. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
