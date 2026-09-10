# -*- coding: utf-8 -*-

# Editado 10/09/2026
# Sincroniza las licitaciones de la Junta de Andalucía con Supabase.
#
# Funcionamiento:
# 1. Consulta el Elasticsearch utilizado por el portal oficial.
# 2. Los resultados se solicitan ordenados por fechaPublicacion DESC.
# 3. Se empiezan a consultar los resultados más recientes.
# 4. Solo se procesan las licitaciones publicadas en los últimos 3 días.
# 5. Cuando aparece una licitación anterior al periodo, se detiene
#    completamente la búsqueda.
# 6. NO se utiliza el total de resultados del buscador.
# 7. Se consulta el detalle de cada expediente.
# 8. Se comprueba que la licitación permita presentar ofertas.
# 9. Se busca primero coincidencia por enlace.
# 10. Si no existe el enlace, se busca por título + órgano.
# 11. Si existe con otra fuente, se añade Andalucía a "fuente".
# 12. Las nuevas licitaciones se marcan como es_novedad=True.
# 13. Las modificaciones reales se marcan como es_actualizada=True.
# 14. Añadir Andalucía como fuente NO se considera una actualización.
# 15. Se genera embedding para las nuevas licitaciones.
# 16. El CPV se obtiene directamente de "codigosCpv" de Elasticsearch
#     y se guarda únicamente el código, sin la denominación.
#
# IMPORTANTE:
# Solo se guardan licitaciones que actualmente permiten presentar ofertas.
# Se descartan estados como Resuelto, Formalizado, Adjudicado, Cerrado,
# Anulado, Desierto, etc., así como licitaciones cuyo plazo ya ha terminado.


from datetime import datetime, date, timedelta
import os
import time
import re

import requests
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURACIÓN DE SUPABASE
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
TAMANO_PAGINA = 100

# Hoy + los dos días anteriores.
#
# Ejemplo si hoy es 10/09/2026:
#   10/09/2026
#   09/09/2026
#   08/09/2026
#
# Cuando aparezca 07/09/2026 o anterior,
# se detiene la búsqueda.
DIAS_ATRAS = 2

# Pausa entre peticiones.
PAUSA_ENTRE_PETICIONES = 0.15

# Tamaño de lote utilizado para cargar Supabase.
TAMANO_LOTE = 500


# ============================================================
# 4. MODELO DE EMBEDDINGS
# ============================================================

MODELO_EMBEDDING = "intfloat/multilingual-e5-small"


# ============================================================
# 5. SESIÓN HTTP
# ============================================================

session = requests.Session()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/152.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Content-Type": "application/json"
}

session.headers.update(HEADERS)


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

    texto = texto.replace("\xa0", " ")
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def normalizar_texto(valor):
    """
    Normaliza texto para realizar comparaciones.
    """

    texto = limpiar_texto(valor).lower()

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

    return normalizar_texto(organo)


def obtener_valor(data, *claves):
    """
    Devuelve el primer valor existente entre las claves indicadas.
    """

    if not isinstance(data, dict):
        return None

    for clave in claves:

        if (
            clave in data
            and data[clave] is not None
        ):
            return data[clave]

    return None


def convertir_importe(valor):
    """
    Convierte importes españoles a float.

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

    texto = limpiar_texto(valor)

    if not texto:
        return None

    formatos = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y"
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

        if texto_iso.endswith("Z"):

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
    Convierte datetime al formato utilizado por Supabase,
    guardando únicamente la fecha, sin hora.
    """

    if fecha is None:
        return None

    if isinstance(
        fecha,
        datetime
    ):
        return fecha.strftime(
            "%Y-%m-%d"
        )

    if isinstance(
        fecha,
        date
    ):
        return fecha.strftime(
            "%Y-%m-%d"
        )

    fecha_convertida = extraer_fecha(
        fecha
    )

    if fecha_convertida is None:
        return None

    return fecha_convertida.strftime(
        "%Y-%m-%d"
    )


# ============================================================
# 7. FILTRO DE ESTADO Y PLAZO
# ============================================================

def estado_permite_presentar_ofertas(estado):
    """
    Devuelve True únicamente si el estado indica que
    la licitación sigue abierta y permite presentar ofertas.

    Los estados desconocidos se rechazan por seguridad.
    """

    estado_norm = normalizar_texto(
        estado
    )

    if not estado_norm:
        return False

    # Estados que indican que ya no se pueden presentar ofertas.
    palabras_bloqueo = (
        "cerrado",
        "resuelto",
        "formalizado",
        "adjudic",
        "desierto",
        "anulado",
        "finalizado",
        "desistimiento",
        "renuncia",
        "cancelado",
        "archivado",
        "suspend"
    )

    if any(
        palabra in estado_norm
        for palabra in palabras_bloqueo
    ):
        return False

    # Solo aceptamos estados claramente abiertos.
    palabras_abierto = (
        "publicada",
        "publicado",
        "en plazo",
        "abierto",
        "plazo de presentacion",
        "licitacion abierta"
    )

    return any(
        palabra in estado_norm
        for palabra in palabras_abierto
    )


def puede_presentar_ofertas(datos):
    """
    Comprueba si la licitación permite actualmente
    presentar ofertas.

    Se exige:
    1. Estado abierto.
    2. Fecha límite existente.
    3. Fecha límite igual o posterior a ahora.
    """

    estado = limpiar_texto(
        datos.get("estado")
    )

    if not estado_permite_presentar_ofertas(
        estado
    ):

        return (
            False,
            "Estado no permite presentar ofertas: "
            f"{estado or 'desconocido'}"
        )

    fecha_fin = datos.get(
        "fecha_fin"
    )

    if fecha_fin is None:

        return (
            False,
            "No tiene fecha límite de presentación"
        )

    ahora = datetime.now()

    if fecha_fin < ahora:

        return (
            False,
            "El plazo de presentación ya ha finalizado"
        )

    return True, ""


# ============================================================
# 8. FUNCIONES PARA LA FUENTE
# ============================================================

def obtener_fuentes(fuente):
    """
    Convierte el campo fuente en una lista.

    Ejemplo:
        "Galicia, TED"
        ->
        ["Galicia", "TED"]
    """

    if not fuente:
        return []

    return [
        limpiar_texto(f)
        for f in str(fuente).split(",")
        if limpiar_texto(f)
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


# ============================================================
# 9. ENLACES
# ============================================================

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
    Construye la URL pública actual del detalle.
    """

    if not codigo:
        return None

    return (
        f"{BASE_URL}/perfiles-licitaciones/"
        f"detalle-licitacion?idExpediente={codigo}"
    )


# ============================================================
# 10. NORMALIZACIÓN DE TIPO DE CONTRATO
# ============================================================

def normalizar_tipo_contrato(tipo):
    """
    Normaliza los tipos de contrato.
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

    tipo_norm = normalizar_texto(
        tipo
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

    return tipo


# ============================================================
# 11. LUGAR DE EJECUCIÓN
# ============================================================

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

    # Elimina "(código NUTS)" o "(codigo NUTS)"
    texto = re.sub(
        r"\(c[oó]digo\s+NUTS\)",
        "",
        texto,
        flags=re.I
    )

    # Elimina códigos NUTS tipo ES618 -
    texto = re.sub(
        r"\bES\d{3}\s*-\s*",
        "",
        texto,
        flags=re.I
    )

    texto = re.sub(
        r"\s*-\s*",
        " - ",
        texto
    )

    return texto.strip() or None


# ============================================================
# 12. CPV
# ============================================================

def extraer_cpv(source):
    """
    Extrae únicamente los códigos CPV del campo
    "codigosCpv" de Elasticsearch.

    Ejemplo:

        "codigosCpv": [
            {
                "codigo": "50850000-8",
                "denominacion":
                    "Servicios de reparación y mantenimiento..."
            }
        ]

    Resultado:

        "50850000-8"

    Si hay varios CPV:

        "50850000-8, 30213100-6"

    La denominación NO se guarda.
    """

    if not isinstance(
        source,
        dict
    ):
        return None

    cpv_raw = source.get(
        "codigosCpv"
    )

    if not isinstance(
        cpv_raw,
        list
    ):
        return None

    codigos = []

    for elemento in cpv_raw:

        if not isinstance(
            elemento,
            dict
        ):
            continue

        codigo = elemento.get(
            "codigo"
        )

        if codigo is None:
            continue

        codigo = limpiar_texto(
            codigo
        )

        # Extraer únicamente:
        # 8 dígitos + guion + 1 dígito
        coincidencias = re.findall(
            r"\b\d{8}-\d\b",
            codigo
        )

        for codigo_extraido in coincidencias:

            if codigo_extraido not in codigos:

                codigos.append(
                    codigo_extraido
                )

    return (
        ", ".join(codigos)
        if codigos
        else None
    )


# ============================================================
# 13. TEXTO RECURSIVO
# ============================================================

def extraer_texto_recursivo(obj):
    """
    Convierte recursivamente una estructura JSON
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
# 14. PETICIONES A ELASTICSEARCH
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
            "ERROR en petición a Elasticsearch: "
            f"{e}"
        )

        return None

    except ValueError as e:

        print(
            "ERROR interpretando la respuesta JSON "
            f"de Elasticsearch: {e}"
        )

        return None


# ============================================================
# 15. OBTENER UNA PÁGINA DEL BUSCADOR
# ============================================================

def obtener_pagina_busqueda(
    desde
):
    """
    Obtiene una página del buscador de Andalucía.

    IMPORTANTE:
    - No se utiliza ningún filtro de fecha en Elasticsearch.
    - Los resultados se ordenan por fechaPublicacion DESC.
    - La búsqueda comienza por los resultados más recientes.
    - La función que llama a esta función se detiene cuando
      encuentra una fecha anterior al periodo.
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
# 16. OBTENER DETALLE
# ============================================================

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
# 17. EXTRAER HITS
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


def obtener_source(hit):
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

    source = obtener_source(
        hit
    )

    codigo = obtener_valor(
        source,
        "idExpediente",
        "id",
        "codigoExpediente"
    )

    if codigo is not None:

        return str(
            codigo
        )

    codigo = hit.get(
        "_id"
    )

    if codigo is not None:

        return str(
            codigo
        )

    return None


# ============================================================
# 18. OBTENER LICITACIONES DE LOS ÚLTIMOS 3 DÍAS
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

    Cuando aparece una fecha anterior a la fecha mínima,
    se detiene completamente la búsqueda.

    NO se utiliza hits.total.
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
    print("BUSCADOR DE ANDALUCÍA")
    print("=" * 70)

    print(
        "Fecha actual: "
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
        "Tamaño de página: "
        f"{TAMANO_PAGINA}"
    )

    print(
        "El total de resultados NO se consulta."
    )

    print("=" * 70)

    while True:

        print()
        print(
            "Consultando resultados "
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

            fecha = fecha_publicacion.date()

            # ------------------------------------------------
            # Fecha anterior al periodo.
            # Como está ordenado DESC, podemos parar.
            # ------------------------------------------------

            if fecha < fecha_minima:

                print()
                print(
                    "Se ha encontrado una licitación "
                    "anterior al periodo:"
                )

                print(
                    f"  Expediente: {codigo}"
                )

                print(
                    "  Fecha publicación: "
                    f"{fecha.strftime('%d/%m/%Y')}"
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
            # Fecha posterior a hoy.
            # Se ignora por seguridad.
            # ------------------------------------------------

            if fecha > hoy:

                continue

            # ------------------------------------------------
            # Está dentro del periodo.
            # ------------------------------------------------

            resultados_periodo.append({
                "hit": hit,
                "codigo": codigo,
                "fecha_publicacion": fecha_publicacion
            })

        if detener:

            break

        # Si recibimos menos resultados de los solicitados,
        # ya no quedan más páginas.

        if len(hits) < TAMANO_PAGINA:

            print()
            print(
                "No quedan más resultados."
            )

            break

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
# 19. CARGAR LICITACIONES EXISTENTES DE SUPABASE
# ============================================================

def cargar_licitaciones_existentes():
    """
    Carga las licitaciones existentes de Supabase.

    Se crean dos índices:
        - por enlace
        - por título + órgano
    """

    print()
    print("=" * 70)
    print("CARGANDO LICITACIONES EXISTENTES DE SUPABASE")
    print("=" * 70)

    registros = []

    inicio = 0

    while True:

        try:

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
                    inicio + TAMANO_LOTE - 1
                )
                .execute()
            )

        except Exception as e:

            print(
                "ERROR cargando Supabase: "
                f"{e}"
            )

            break

        lote = respuesta.data or []

        if not lote:
            break

        registros.extend(
            lote
        )

        if len(lote) < TAMANO_LOTE:
            break

        inicio += TAMANO_LOTE

    por_enlace = {}
    por_titulo_organo = {}

    for registro in registros:

        enlace = limpiar_texto(
            registro.get("enlace")
        )

        if enlace:

            por_enlace[
                enlace
            ] = registro

        titulo = limpiar_texto(
            registro.get("titulo")
        )

        organo = normalizar_organo(
            registro.get("organo")
        )

        if titulo:

            clave = (
                normalizar_texto(titulo),
                organo
            )

            por_titulo_organo[
                clave
            ] = registro

    print(
        "Registros cargados de Supabase: "
        f"{len(registros)}"
    )

    print("=" * 70)

    return (
        registros,
        por_enlace,
        por_titulo_organo
    )


# ============================================================
# 20. GENERAR EMBEDDING
# ============================================================

def generar_embedding(
    registro,
    modelo
):
    """
    Genera embedding para una licitación nueva.
    """

    partes = []

    if registro.get(
        "titulo"
    ):

        partes.append(
            f"Título: {registro['titulo']}"
        )

    if registro.get(
        "organo"
    ):

        partes.append(
            f"Órgano: {registro['organo']}"
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
            f"CPV: {registro['cpv']}"
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
            registro["texto_completo"]
        )

    texto = "\n".join(
        partes
    )

    if not texto.strip():
        return None

    try:

        embedding = modelo.encode(
            texto,
            normalize_embeddings=True
        )

        return embedding.tolist()

    except Exception as e:

        print(
            "  [ERROR] Generando embedding: "
            f"{e}"
        )

        return None


# ============================================================
# 21. COMPARAR CAMBIOS
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
        and valor_antiguo is None
    ):
        return False

    if (
        valor_nuevo is None
        and (
            valor_antiguo is None
            or str(valor_antiguo).strip() == ""
        )
    ):
        return False

    if (
        valor_antiguo is None
        and (
            valor_nuevo is None
            or str(valor_nuevo).strip() == ""
        )
    ):
        return False

    if isinstance(
        valor_nuevo,
        (int, float)
    ):

        try:

            return (
                abs(
                    valor_nuevo
                    - float(valor_antiguo)
                )
                > 0.000001
            )

        except (
            TypeError,
            ValueError
        ):

            return True

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
    Comprueba si han cambiado datos reales.

    El campo "fuente" no se considera aquí.
    """

    campos = [
        "titulo",
        "organo",
        "fecha",
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

        if campo in {
            "fecha",
            "fecha_fin"
        }:

            valor_nuevo = formatear_fecha_supabase(
                valor_nuevo
            )

            valor_antiguo = formatear_fecha_supabase(
                valor_antiguo
            )

        if valores_diferentes(
            valor_nuevo,
            valor_antiguo
        ):

            return True

    return False


# ============================================================
# 22. PREPARAR REGISTRO PARA SUPABASE
# ============================================================

def preparar_registro_supabase(
    datos,
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
        "fecha": formatear_fecha_supabase(
            datos.get("fecha")
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
        "fecha_fin": formatear_fecha_supabase(
            datos.get("fecha_fin")
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

        registro["embedding"] = embedding

    return registro


# ============================================================
# 23. EXTRAER DATOS DEL DETALLE
# ============================================================

def extraer_datos_detalle(
    respuesta,
    codigo_expediente,
    fecha_publicacion
):
    """
    Extrae los datos principales del detalle.
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

    cpv = extraer_cpv(
        source
    )

    # --------------------------------------------------------
    # Fecha límite de presentación
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
    # Estado
    # --------------------------------------------------------

    estado = obtener_valor(
        source,
        "estado"
    )

    if isinstance(
        estado,
        dict
    ):

        estado = obtener_valor(
            estado,
            "nombre",
            "descripcion",
            "description",
            "name"
        )

    estado = (
        limpiar_texto(estado)
        or None
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
        "titulo": (
            limpiar_texto(titulo)
            or None
        ),
        "organo": (
            limpiar_texto(organo)
            or None
        ),
        "fecha": fecha_publicacion,
        "importe": importe,
        "tipo_contrato": tipo_contrato,
        "cpv": cpv,
        "fecha_fin": fecha_fin,
        "lugar_ejecucion": lugar,
        "estado": estado,
        "texto_completo": (
            texto_completo
            if texto_completo
            else None
        )
    }


# ============================================================
# 24. PROCESAR UNA LICITACIÓN
# ============================================================

def procesar_licitacion(
    datos,
    por_enlace,
    por_titulo_organo,
    modelo,
    contadores
):
    """
    Inserta o actualiza una licitación.
    """

    enlace = limpiar_texto(
        datos.get("enlace")
    )

    titulo = limpiar_texto(
        datos.get("titulo")
    )

    organo = limpiar_texto(
        datos.get("organo")
    )

    if not enlace:

        contadores["errores"] += 1

        print(
            "  [ERROR] La licitación no tiene enlace."
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

    if existente is None and titulo:

        clave = (
            normalizar_texto(titulo),
            normalizar_organo(organo)
        )

        existente = (
            por_titulo_organo.get(
                clave
            )
        )

        if existente is not None:

            contadores["duplicadas"] += 1

            print(
                "  [DUPLICADA] Coincidencia "
                "por título + órgano."
            )

    # ========================================================
    # NUEVA
    # ========================================================

    if existente is None:

        print(
            "  [NUEVA] No existe en Supabase."
        )

        embedding = generar_embedding(
            datos,
            modelo
        )

        registro = preparar_registro_supabase(
            datos,
            embedding
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
                respuesta.data
                or []
            )

            if not insertados:

                contadores["errores"] += 1

                print(
                    "  [ERROR] Supabase no devolvió "
                    "el registro insertado."
                )

                return

            nuevo_registro = insertados[0]

            por_enlace[
                enlace
            ] = nuevo_registro

            clave = (
                normalizar_texto(titulo),
                normalizar_organo(organo)
            )

            if titulo:

                por_titulo_organo[
                    clave
                ] = nuevo_registro

            contadores["nuevas"] += 1

            print(
                "  [INSERTADA] Licitación nueva."
            )

        except Exception as e:

            contadores["errores"] += 1

            print(
                "  [ERROR] Insertando en Supabase: "
                f"{e}"
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

    # --------------------------------------------------------
    # Actualizar datos reales
    # --------------------------------------------------------

    if cambios:

        campos_actualizables = [
            "titulo",
            "organo",
            "fecha",
            "importe",
            "tipo_contrato",
            "cpv",
            "fecha_fin",
            "lugar_ejecucion",
            "texto_completo"
        ]

        for campo in campos_actualizables:

            valor = datos.get(
                campo
            )

            if campo in {
                "fecha",
                "fecha_fin"
            }:

                valor = formatear_fecha_supabase(
                    valor
                )

            actualizacion[
                campo
            ] = valor

        actualizacion[
            "es_actualizada"
        ] = True

        contadores["actualizadas"] += 1

        print(
            "  [ACTUALIZADA] Han cambiado "
            "datos reales."
        )

    # --------------------------------------------------------
    # Añadir Andalucía como fuente
    # --------------------------------------------------------

    if not tiene_andalucia:

        nueva_fuente = añadir_fuente(
            fuente_actual,
            FUENTE
        )

        actualizacion[
            "fuente"
        ] = nueva_fuente

        print(
            "  [FUENTE] Añadida Andalucía."
        )

    # --------------------------------------------------------
    # Si solo se añadió la fuente,
    # NO se marca como actualización.
    # --------------------------------------------------------

    if not actualizacion:

        contadores["existentes"] += 1

        print(
            "  [SIN CAMBIOS] Ya existe."
        )

        return

    # ========================================================
    # ACTUALIZAR SUPABASE
    # ========================================================

    try:

        (
            supabase
            .table("licitaciones")
            .update(actualizacion)
            .eq(
                "id",
                existente["id"]
            )
            .execute()
        )

        # Actualizar también los índices en memoria.

        existente_actualizado = dict(
            existente
        )

        existente_actualizado.update(
            actualizacion
        )

        por_enlace[
            enlace
        ] = existente_actualizado

        clave = (
            normalizar_texto(titulo),
            normalizar_organo(organo)
        )

        if titulo:

            por_titulo_organo[
                clave
            ] = existente_actualizado

    except Exception as e:

        contadores["errores"] += 1

        print(
            "  [ERROR] Actualizando Supabase: "
            f"{e}"
        )


# ============================================================
# 25. MAIN
# ============================================================

def main():

    inicio_tiempo = time.time()

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
            "para procesar."
        )

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
    # CARGAR MODELO
    # ========================================================

    print()
    print("=" * 70)
    print("CARGANDO MODELO DE EMBEDDINGS")
    print("=" * 70)

    try:

        modelo = SentenceTransformer(
            MODELO_EMBEDDING
        )

        print(
            "Modelo cargado correctamente."
        )

    except Exception as e:

        print(
            "ERROR cargando modelo de embeddings: "
            f"{e}"
        )

        return

    # ========================================================
    # CONTADORES
    # ========================================================

    contadores = {
        "nuevas": 0,
        "actualizadas": 0,
        "existentes": 0,
        "duplicadas": 0,
        "errores": 0,
        "descartadas_no_presentables": 0
    }

    # ========================================================
    # PROCESAR RESULTADOS
    # ========================================================

    print()
    print("=" * 70)
    print("PROCESANDO LICITACIONES")
    print("=" * 70)

    procesadas = set()

    for indice, resultado in enumerate(
        resultados,
        start=1
    ):

        hit = resultado.get(
            "hit"
        )

        codigo = resultado.get(
            "codigo"
        )

        fecha_publicacion = resultado.get(
            "fecha_publicacion"
        )

        if not codigo:

            contadores["errores"] += 1

            print(
                f"[{indice}/{len(resultados)}] "
                "Sin identificador de expediente."
            )

            continue

        # ====================================================
        # EVITAR DUPLICADOS EN LA PROPIA RESPUESTA
        # ====================================================

        if codigo in procesadas:

            contadores["duplicadas"] += 1

            print(
                f"[{indice}/{len(resultados)}] "
                f"Expediente {codigo}: "
                "duplicado en la consulta."
            )

            continue

        procesadas.add(
            codigo
        )

        print()
        print(
            f"[{indice}/{len(resultados)}] "
            f"Expediente: {codigo}"
        )

        # Fecha SIN hora
        print(
            "    Fecha publicación: "
            f"{fecha_publicacion.strftime('%Y-%m-%d')}"
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

            contadores["errores"] += 1

            print(
                "    [ERROR] No se pudo obtener "
                "el detalle."
            )

            continue

        # ====================================================
        # EXTRAER DATOS
        # ====================================================

        datos = (
            extraer_datos_detalle(
                respuesta_detalle,
                codigo,
                fecha_publicacion
            )
        )

        if datos is None:

            contadores["errores"] += 1

            print(
                "    [ERROR] No se pudieron extraer "
                "los datos del detalle."
            )

            continue

        # ====================================================
        # MOSTRAR INFORMACIÓN PRINCIPAL
        # ====================================================

        print(
            "    Título: "
            f"{datos.get('titulo')}"
        )

        print(
            "    Órgano: "
            f"{datos.get('organo')}"
        )

        print(
            "    Estado: "
            f"{datos.get('estado')}"
        )

        print(
            "    Tipo contrato: "
            f"{datos.get('tipo_contrato')}"
        )

        print(
            "    Importe: "
            f"{datos.get('importe')}"
        )

        print(
            "    CPV: "
            f"{datos.get('cpv')}"
        )

        print(
            "    Fecha fin: "
            f"{formatear_fecha_supabase(datos.get('fecha_fin'))}"
        )

        print(
            "    Lugar: "
            f"{datos.get('lugar_ejecucion')}"
        )

        print(
            "    Enlace: "
            f"{datos.get('enlace')}"
        )

        # ====================================================
        # COMPROBAR SI SE PUEDEN PRESENTAR OFERTAS
        # ====================================================

        puede_presentar, motivo = (
            puede_presentar_ofertas(
                datos
            )
        )

        if not puede_presentar:

            contadores[
                "descartadas_no_presentables"
            ] += 1

            print(
                "    [DESCARTADA] "
                f"{motivo}"
            )

            continue

        print(
            "    [VÁLIDA] Permite presentar ofertas."
        )

        # ====================================================
        # SINCRONIZAR
        # ====================================================

        procesar_licitacion(
            datos,
            por_enlace,
            por_titulo_organo,
            modelo,
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
        - inicio_tiempo
    )

    print()
    print("=" * 70)
    print("RESUMEN DE SINCRONIZACIÓN")
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
        "Duplicadas: "
        f"{contadores['duplicadas']}"
    )

    print(
        "Descartadas por estado/plazo: "
        f"{contadores['descartadas_no_presentables']}"
    )

    print(
        "Errores: "
        f"{contadores['errores']}"
    )

    print(
        "Tiempo total: "
        f"{tiempo_total:.2f} segundos"
    )

    print("=" * 70)
    print(
        "SINCRONIZACIÓN FINALIZADA"
    )
    print("=" * 70)


# ============================================================
# 26. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
