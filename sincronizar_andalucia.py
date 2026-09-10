# -*- coding: utf-8 -*-

# Editado 10/09/2026
#
# Sincroniza las licitaciones de Andalucía con Supabase.
#
# Funcionamiento:
# 1. Consulta el buscador Elasticsearch de Andalucía.
# 2. Los resultados se solicitan ordenados por fechaPublicacion DESC.
# 3. Se empiezan a consultar los resultados más recientes.
# 4. Solo se procesan las licitaciones publicadas en los últimos 3 días.
# 5. Cuando aparece una licitación anterior al periodo,
#    se detiene completamente la búsqueda.
# 6. NO se utiliza el total de resultados del buscador.
# 7. Se buscan coincidencias por enlace.
# 8. Si no existe el enlace, se busca por título + órgano.
# 9. Si existe con otra fuente, se añade Andalucía a "fuente".
# 10. Las nuevas licitaciones se marcan como es_novedad=True.
# 11. Las modificaciones reales se marcan como es_actualizada=True.
# 12. Añadir Andalucía como fuente no se considera una actualización.
# 13. Se genera embedding para las nuevas licitaciones.


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


# ============================================================
# 2. CONFIGURACIÓN DE ANDALUCÍA
# ============================================================

FUENTE = "Andalucía"

BASE_URL = (
    "https://www.juntadeandalucia.es/"
    "haciendayadministracionpublica/apl/pdc-front-publico"
)

# Buscador general de expedientes
URL_BUSCADOR = (
    f"{BASE_URL}/elastic/"
    "sirec_pdc_expedientes/_search?pretty"
)

# Índice de detalle de expedientes
URL_DETALLE = (
    f"{BASE_URL}/elastic/"
    "sirec_pdc_expedientes_details/_search?pretty"
)


# ============================================================
# 3. PARÁMETROS
# ============================================================

# Número de resultados solicitados en cada petición.
#
# Como el portal está ordenado por fechaPublicacion DESC,
# 100 resultados permiten avanzar mucho más rápido.
TAMANO_PAGINA = 100

# Últimos 3 días:
#
# Si hoy es 10/09/2026:
#   10/09/2026
#   09/09/2026
#   08/09/2026
#
# Cuando aparezca 07/09/2026 o anterior,
# se detiene la búsqueda.
DIAS_ATRAS = 2

# Pausa entre peticiones
PAUSA_ENTRE_PETICIONES = 0.15

# Tamaño de lote para cargar Supabase
TAMANO_LOTE = 500


# ============================================================
# 4. MODELO DE EMBEDDINGS
# ============================================================

MODELO_EMBEDDING = "intfloat/multilingual-e5-small"

print("Cargando modelo de embeddings...")

model = SentenceTransformer(
    MODELO_EMBEDDING,
    device="cpu"
)

print("Modelo de embeddings cargado.")


# ============================================================
# 5. SESIÓN HTTP
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
    )
})


# ============================================================
# 6. FUNCIONES AUXILIARES
# ============================================================

def limpiar_texto(valor):
    """
    Limpia espacios y caracteres especiales.
    """

    if valor is None:
        return ""

    texto = str(valor)

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


def normalizar_texto(valor):
    """
    Normaliza texto para realizar comparaciones.
    """

    texto = limpiar_texto(
        valor
    ).lower()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n"
    }

    for origen, destino in reemplazos.items():

        texto = texto.replace(
            origen,
            destino
        )

    return texto


def normalizar_organo(organo):
    """
    Normaliza el órgano de contratación.
    """

    return normalizar_texto(
        organo
    )


def obtener_valor(data, *claves):
    """
    Devuelve el primer valor existente entre las claves.
    """

    if not isinstance(
        data,
        dict
    ):
        return None

    for clave in claves:

        if clave in data:

            valor = data[clave]

            if valor is not None and valor != "":
                return valor

    return None


def extraer_fecha(valor):
    """
    Convierte distintos formatos de fecha a datetime.
    """

    if valor is None:
        return None

    if isinstance(
        valor,
        datetime
    ):
        return valor

    if isinstance(
        valor,
        date
    ):
        return datetime.combine(
            valor,
            datetime.min.time()
        )

    texto = limpiar_texto(
        valor
    )

    if not texto:
        return None

    formatos = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
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

            fecha = datetime.strptime(
                texto,
                formato
            )

            if fecha.tzinfo is not None:

                fecha = fecha.replace(
                    tzinfo=None
                )

            return fecha

        except ValueError:
            continue

    try:

        texto_iso = texto

        if (
            texto_iso.endswith("Z")
        ):
            texto_iso = (
                texto_iso[:-1]
                + "+00:00"
            )

        # Convierte +0200 en +02:00
        texto_iso = re.sub(
            r"([+-]\d{2})(\d{2})$",
            r"\1:\2",
            texto_iso
        )

        fecha = datetime.fromisoformat(
            texto_iso
        )

        if fecha.tzinfo is not None:

            fecha = fecha.replace(
                tzinfo=None
            )

        return fecha

    except ValueError:
        return None


def formatear_fecha_supabase(fecha):
    """
    Convierte datetime al formato utilizado por Supabase.
    """

    if fecha is None:
        return None

    if isinstance(
        fecha,
        date
    ) and not isinstance(
        fecha,
        datetime
    ):

        fecha = datetime.combine(
            fecha,
            datetime.min.time()
        )

    return fecha.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def convertir_importe(valor):
    """
    Convierte un importe español a float.

    Ejemplo:
        6.021.000,00 € -> 6021000.0
    """

    if valor is None:
        return None

    if isinstance(
        valor,
        (int, float)
    ):
        return float(valor)

    texto = limpiar_texto(
        valor
    )

    if not texto:
        return None

    texto = texto.replace(
        "€",
        ""
    )

    texto = texto.replace(
        "EUR",
        ""
    )

    texto = texto.strip()

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

        texto = texto.replace(
            " ",
            ""
        )

    texto = re.sub(
        r"[^\d.\-]",
        "",
        texto
    )

    if not texto:
        return None

    try:

        return float(
            texto
        )

    except ValueError:

        return None


def obtener_fuentes(fuente):
    """
    Convierte el campo fuente en una lista.
    """

    if not fuente:
        return []

    return [
        parte.strip()
        for parte in str(fuente).split(",")
        if parte.strip()
    ]


def contiene_fuente(
    fuente,
    fuente_buscar
):
    """
    Comprueba si una fuente está incluida.
    """

    fuentes = obtener_fuentes(
        fuente
    )

    for fuente_actual in fuentes:

        if (
            normalizar_texto(
                fuente_actual
            )
            ==
            normalizar_texto(
                fuente_buscar
            )
        ):
            return True

    return False


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

    if not contiene_fuente(
        fuente_actual,
        nueva_fuente
    ):

        fuentes.append(
            nueva_fuente
        )

    return ", ".join(
        fuentes
    )


def normalizar_tipo_contrato(tipo):
    """
    Normaliza el tipo de contrato.
    """

    if isinstance(
        tipo,
        dict
    ):

        tipo = obtener_valor(
            tipo,
            "descripcion",
            "description",
            "nombre",
            "name"
        )

    tipo = limpiar_texto(
        tipo
    )

    if not tipo:
        return None

    return tipo


def limpiar_lugar_ejecucion(lugar):
    """
    Limpia el lugar de ejecución.
    """

    if lugar is None:
        return None

    if isinstance(
        lugar,
        list
    ):

        valores = []

        for elemento in lugar:

            limpio = limpiar_lugar_ejecucion(
                elemento
            )

            if limpio:
                valores.append(
                    limpio
                )

        resultado = []

        for valor in valores:

            if valor not in resultado:

                resultado.append(
                    valor
                )

        return (
            ", ".join(resultado)
            if resultado
            else None
        )

    if isinstance(
        lugar,
        dict
    ):

        lugar = obtener_valor(
            lugar,
            "descripcion",
            "description",
            "nombre",
            "name",
            "municipio",
            "provincia"
        )

    texto = limpiar_texto(
        lugar
    )

    if not texto:
        return None

    # Elimina códigos NUTS como ES618
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


def extraer_cpv(cpv):
    """
    Extrae el CPV de diferentes estructuras.
    """

    if cpv is None:
        return None

    if isinstance(
        cpv,
        list
    ):

        valores = []

        for elemento in cpv:

            valor = extraer_cpv(
                elemento
            )

            if valor:
                valores.append(
                    valor
                )

        resultado = []

        for valor in valores:

            if valor not in resultado:

                resultado.append(
                    valor
                )

        return (
            ", ".join(resultado)
            if resultado
            else None
        )

    if isinstance(
        cpv,
        dict
    ):

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


# ============================================================
# 7. PETICIÓN A ELASTICSEARCH
# ============================================================

def consultar_elasticsearch(
    url,
    payload,
    timeout=60
):
    """
    Realiza una petición POST a Elasticsearch.
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
            "ERROR interpretando la respuesta "
            f"JSON: {e}"
        )

        return None


# ============================================================
# 8. OBTENER UNA PÁGINA DEL BUSCADOR
# ============================================================

def obtener_pagina_busqueda(
    desde
):
    """
    Obtiene una página del buscador de Andalucía.

    IMPORTANTE:
    No se utiliza ningún filtro de fecha en Elasticsearch.

    Los resultados se ordenan por fechaPublicacion DESC.
    Por tanto, empezamos por los más recientes y vamos avanzando
    hasta encontrar una fecha anterior al periodo.
    """

    payload = {
        "query": {
            "bool": {
                "must": [],
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
        "track_total_hits": False,
        "from": desde
    }

    return consultar_elasticsearch(
        URL_BUSCADOR,
        payload
    )


# ============================================================
# 9. OBTENER DETALLE
# ============================================================

def obtener_detalle(
    codigo_expediente
):
    """
    Obtiene el detalle de un expediente.
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
# 10. EXTRAER HITS
# ============================================================

def extraer_hits_busqueda(
    respuesta
):
    """
    Extrae los resultados de una respuesta Elasticsearch.
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


def obtener_source(
    hit
):
    """
    Obtiene el _source de un hit.
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


def extraer_codigo_hit(
    hit
):
    """
    Extrae el identificador del expediente.
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
        "codigoExpediente",
        "numeroExpediente"
    )

    if codigo:

        return str(
            codigo
        )

    return None


def extraer_fecha_publicacion_hit(
    hit
):
    """
    Extrae fechaPublicacion del resultado.
    """

    source = obtener_source(
        hit
    )

    valor = obtener_valor(
        source,
        "fechaPublicacion"
    )

    return extraer_fecha(
        valor
    )


# ============================================================
# 11. OBTENER LICITACIONES DE LOS ÚLTIMOS 3 DÍAS
# ============================================================

def obtener_licitaciones_ultimos_tres_dias():
    """
    Obtiene las licitaciones publicadas en los últimos 3 días.

    El buscador está ordenado por fechaPublicacion DESC.

    Ejemplo:

        10/09/2026
        10/09/2026
        09/09/2026
        08/09/2026
        08/09/2026
        07/09/2026  <-- PARAR

    En cuanto aparece una fecha anterior a la fecha mínima,
    se detiene completamente la consulta.
    """

    hoy = date.today()

    fecha_minima = (
        hoy
        - timedelta(
            days=DIAS_ATRAS
        )
    )

    resultados_periodo = []

    desde = 0

    print()
    print("=" * 70)
    print(
        "BUSCADOR DE ANDALUCÍA"
    )
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

    print(
        "Orden: fechaPublicacion descendente"
    )

    print(
        "El total de resultados NO se consulta."
    )

    print("=" * 70)

    while True:

        print()
        print(
            f"Consultando resultados "
            f"{desde + 1}-"
            f"{desde + TAMANO_PAGINA}..."
        )

        respuesta = obtener_pagina_busqueda(
            desde
        )

        if respuesta is None:

            print(
                "No se pudo obtener la página."
            )

            break

        hits = extraer_hits_busqueda(
            respuesta
        )

        if not hits:

            print(
                "No quedan más resultados."
            )

            break

        detener = False

        for hit in hits:

            codigo = extraer_codigo_hit(
                hit
            )

            fecha_publicacion = (
                extraer_fecha_publicacion_hit(
                    hit
                )
            )

            if not codigo:

                continue

            if fecha_publicacion is None:

                print(
                    f"  [AVISO] Expediente {codigo} "
                    "sin fecha de publicación."
                )

                continue

            fecha = (
                fecha_publicacion.date()
            )

            # ------------------------------------------------
            # Si es anterior al periodo:
            # PARAR COMPLETAMENTE.
            # ------------------------------------------------

            if fecha < fecha_minima:

                print()
                print(
                    "Se ha encontrado una licitación "
                    f"anterior al periodo "
                    f"({fecha.strftime('%d/%m/%Y')})."
                )

                print(
                    "Como los resultados están ordenados "
                    "de más reciente a más antiguo,"
                )

                print(
                    "se detiene completamente la búsqueda."
                )

                detener = True

                break

            # ------------------------------------------------
            # Si es posterior a hoy, se ignora.
            # ------------------------------------------------

            if fecha > hoy:

                continue

            # ------------------------------------------------
            # Está dentro del periodo.
            # ------------------------------------------------

            resultados_periodo.append({
                "hit": hit,
                "codigo": codigo,
                "fecha_publicacion": (
                    fecha_publicacion
                )
            })

        # ----------------------------------------------------
        # Si hemos encontrado una fecha antigua,
        # terminamos todo el proceso.
        # ----------------------------------------------------

        if detener:

            break

        # ----------------------------------------------------
        # Si hemos recibido menos resultados que los pedidos,
        # no quedan más.
        # ----------------------------------------------------

        if len(hits) < TAMANO_PAGINA:

            print()
            print(
                "No quedan más resultados."
            )

            break

        # ----------------------------------------------------
        # Pasar a la siguiente página.
        # ----------------------------------------------------

        desde += TAMANO_PAGINA

        time.sleep(
            PAUSA_ENTRE_PETICIONES
        )

    print()
    print("=" * 70)

    print(
        "Licitaciones recuperadas dentro "
        "de los últimos 3 días: "
        f"{len(resultados_periodo)}"
    )

    print("=" * 70)

    return resultados_periodo


# ============================================================
# 12. CARGAR SUPABASE
# ============================================================

def cargar_licitaciones_existentes():
    """
    Carga las licitaciones existentes de Supabase.
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

            por_enlace[
                enlace
            ] = registro

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
# 13. GENERAR EMBEDDING
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
            "Lugar de ejecución: "
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
# 14. COMPARAR CAMBIOS
# ============================================================

def valores_diferentes(
    valor_nuevo,
    valor_antiguo
):
    """
    Comprueba si dos valores son diferentes.
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

        return (
            abs(
                valor_nuevo
                - float(valor_antiguo)
            )
            > 0.000001
        )

    return (
        str(
            valor_nuevo
        ).strip()
        !=
        str(
            valor_antiguo
        ).strip()
    )


def hay_cambios_reales(
    nuevo,
    existente
):
    """
    Comprueba si han cambiado datos reales.

    El campo "fuente" no se considera aquí.
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
# 15. EXTRAER DATOS DEL DETALLE
# ============================================================

def extraer_datos_detalle(
    respuesta,
    codigo_expediente
):
    """
    Extrae los datos principales del detalle.

    Se contemplan diferentes nombres de campos para hacer
    la extracción más tolerante a cambios del portal.
    """

    if not respuesta:

        return None

    hits = (
        respuesta
        .get(
            "hits",
            {}
        )
        .get(
            "hits",
            []
        )
    )

    if not hits:

        return None

    hit = hits[0]

    source = hit.get(
        "_source",
        {}
    )

    if not isinstance(
        source,
        dict
    ):

        return None

    # --------------------------------------------------------
    # Título
    # --------------------------------------------------------

    titulo = obtener_valor(
        source,
        "titulo",
        "tituloExpediente",
        "nombre"
    )

    # --------------------------------------------------------
    # Órgano
    # --------------------------------------------------------

    organo = obtener_valor(
        source,
        "organo",
        "organoContratacion",
        "perfilContratante"
    )

    if isinstance(
        organo,
        dict
    ):

        organo = obtener_valor(
            organo,
            "descripcion",
            "description",
            "nombre"
        )

    # --------------------------------------------------------
    # Tipo de contrato
    # --------------------------------------------------------

    tipo_contrato = obtener_valor(
        source,
        "tipoContrato"
    )

    tipo_contrato = normalizar_tipo_contrato(
        tipo_contrato
    )

    # --------------------------------------------------------
    # Importe
    # --------------------------------------------------------

    importe = obtener_valor(
        source,
        "importeLicitacion",
        "importe",
        "importeSinIVA",
        "importeLicitacionSinIVA"
    )

    importe = convertir_importe(
        importe
    )

    # --------------------------------------------------------
    # CPV
    # --------------------------------------------------------

    cpv = obtener_valor(
        source,
        "cpv",
        "clasificacionCPV",
        "clasificacionCpv",
        "codigoCPV"
    )

    cpv = extraer_cpv(
        cpv
    )

    # --------------------------------------------------------
    # Fecha límite
    # --------------------------------------------------------

    fecha_fin = obtener_valor(
        source,
        "fechaLimitePresentacion",
        "fechaFin",
        "fechaFinPresentacion",
        "fechaLimite"
    )

    fecha_fin = extraer_fecha(
        fecha_fin
    )

    # --------------------------------------------------------
    # Lugar
    # --------------------------------------------------------

    lugar = obtener_valor(
        source,
        "lugarEjecucion",
        "lugar",
        "ubicacion"
    )

    lugar = limpiar_lugar_ejecucion(
        lugar
    )

    # --------------------------------------------------------
    # Enlace
    # --------------------------------------------------------

    enlace = construir_enlace(
        codigo_expediente
    )

    # --------------------------------------------------------
    # Texto completo
    # --------------------------------------------------------

    partes_texto = (
        extraer_texto_recursivo(
            source
        )
    )

    texto_completo = "\n".join(
        partes_texto
    )

    return {
        "enlace": enlace,
        "titulo": limpiar_texto(
            titulo
        ) or None,
        "organo": limpiar_texto(
            organo
        ) or None,
        "importe": importe,
        "tipo_contrato": tipo_contrato,
        "cpv": cpv,
        "fecha_fin": fecha_fin,
        "lugar_ejecucion": lugar,
        "texto_completo": (
            texto_completo
            if texto_completo
            else None
        )
    }


# ============================================================
# 16. PREPARAR REGISTRO
# ============================================================

def preparar_registro_supabase(
    datos,
    fecha_publicacion,
    embedding=None
):
    """
    Prepara una nueva licitación para Supabase.
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
# 17. PROCESAR UNA LICITACIÓN
# ============================================================

def procesar_licitacion(
    datos,
    fecha_publicacion,
    por_enlace,
    por_titulo_organo,
    contadores
):
    """
    Inserta o actualiza una licitación.
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

    # ========================================================
    # BUSCAR POR ENLACE
    # ========================================================

    existente = por_enlace.get(
        enlace
    )

    # ========================================================
    # BUSCAR POR TÍTULO + ÓRGANO
    # ========================================================

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
                "  [DUPLICADA] "
                "Coincidencia por título + órgano."
            )

    # ========================================================
    # NUEVA
    # ========================================================

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

    # ========================================================
    # EXISTENTE
    # ========================================================

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

    # ========================================================
    # AÑADIR FUENTE
    # ========================================================

    if not tiene_andalucia:

        actualizacion[
            "fuente"
        ] = añadir_fuente(
            fuente_actual,
            FUENTE
        )

    # ========================================================
    # MARCAR ACTUALIZACIÓN
    # ========================================================

    if cambios:

        actualizacion[
            "es_actualizada"
        ] = True

    # ========================================================
    # SI NO HAY NADA QUE CAMBIAR
    # ========================================================

    if not actualizacion:

        contadores[
            "existentes"
        ] += 1

        print(
            "  [EXISTENTE] Sin cambios."
        )

        return

    # ========================================================
    # ACTUALIZAR SUPABASE
    # ========================================================

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
# 18. EXTRAER TEXTO RECURSIVO
# ============================================================

def extraer_texto_recursivo(
    obj
):
    """
    Convierte recursivamente el JSON del detalle
    en texto para texto_completo.
    """

    partes = []

    if obj is None:

        return partes

    if isinstance(
        obj,
        dict
    ):

        for clave, valor in obj.items():

            if clave in {
                "embedding",
                "vector"
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

    elif isinstance(
        obj,
        list
    ):

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
# 19. MAIN
# ============================================================

def main():

    inicio_total = time.time()

    print()
    print("=" * 70)
    print(
        "SINCRONIZACIÓN DE LICITACIONES - ANDALUCÍA"
    )
    print("=" * 70)

    # ========================================================
    # OBTENER LICITACIONES RECIENTES
    # ========================================================

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

    # ========================================================
    # CARGAR SUPABASE
    # ========================================================

    (
        registros_existentes,
        por_enlace,
        por_titulo_organo
    ) = cargar_licitaciones_existentes()

    # ========================================================
    # CONTADORES
    # ========================================================

    contadores = {
        "nuevas": 0,
        "actualizadas": 0,
        "existentes": 0,
        "duplicadas_titulo_organo": 0,
        "errores": 0
    }

    # ========================================================
    # PROCESAR RESULTADOS
    # ========================================================

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

        # Evitar duplicados dentro de la respuesta
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

        # ====================================================
        # OBTENER DETALLE
        # ====================================================

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

        # ====================================================
        # EXTRAER DATOS
        # ====================================================

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

        # ====================================================
        # SINCRONIZAR
        # ====================================================

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

    # ========================================================
    # RESUMEN
    # ========================================================

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
        "Licitaciones publicadas en los "
        "últimos 3 días: "
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
# 20. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
