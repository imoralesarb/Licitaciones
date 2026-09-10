from datetime import datetime, date, timedelta
import os
import time
import re
import requests
from bs4 import BeautifulSoup
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

print("Cargando modelo de IA (multilingual-e5-small)...")

encoder = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def limpiar_importe_html(texto):
    """
    Limpia cadenas de texto de importes forzando estrictamente
    el formato europeo:

        Punto (.) = separador de miles
        Coma (,) = separador decimal

    Ejemplos:
        '19.880,50' -> 19880.5
        '8.000'     -> 8000.0
        '7.232,2'   -> 7232.2
    """

    try:
        if not texto:
            return 0.0

        if isinstance(texto, (int, float)):
            return float(texto)

        texto_limpio = re.sub(
            r'[^\d,\.-]',
            '',
            str(texto)
        ).strip()

        if not texto_limpio:
            return 0.0

        if ',' in texto_limpio:
            texto_limpio = texto_limpio.replace('.', '')
            texto_limpio = texto_limpio.replace(',', '.')

        elif '.' in texto_limpio:
            partes = texto_limpio.split('.')

            if len(partes[-1]) == 3 and len(partes) > 1:
                texto_limpio = texto_limpio.replace('.', '')

        return float(texto_limpio)

    except ValueError:
        return 0.0


def procesar_lugar_euskadi(lugar_raw):
    """Añade País Vasco al lugar de ejecución detectado."""

    lugar_limpio = (
        str(lugar_raw).strip()
        if lugar_raw
        else "No especificado"
    )

    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "País Vasco"

    lugar_lower = lugar_limpio.lower()

    if (
        "país vasco" not in lugar_lower
        and "euskadi" not in lugar_lower
    ):
        return f"{lugar_limpio}, País Vasco"

    return lugar_limpio


def normalizar_organo(org):
    """
    Extrae la raíz del órgano eliminando subcategorías
    tras guiones o sufijos.
    """

    if not org:
        return ""

    org_limpio = (
        org.split("-")[0]
        .split("—")[0]
        .strip()
        .lower()
    )

    return org_limpio


# ============================================================
# SINCRONIZACIÓN
# ============================================================

def sincronizar_licitaciones_euskadi():

    hoy_date = datetime.now().date()
    ayer_date = hoy_date - timedelta(days=1)

    base_url = "https://api.euskadi.eus/administration/events"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    tipo_licitacion_id = "1"

    fecha_actual = ayer_date
    fecha_fin_rango = hoy_date

    results = []

    print(
        f"Consultando la API de Euskadi por fechas "
        f"del {fecha_actual} al {fecha_fin_rango}..."
    )

    # ========================================================
    # 1. DESCARGAR REGISTROS DE EUSKADI
    # ========================================================

    while fecha_actual <= fecha_fin_rango:

        year = fecha_actual.year
        month = fecha_actual.month
        day = fecha_actual.day

        url = (
            f"{base_url}/v1.0/events/byType/"
            f"{tipo_licitacion_id}/byDate/"
            f"{year}/{month}/{day}"
        )

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            if response.status_code == 200:

                data = response.json()

                events = (
                    data.get("events", [])
                    if isinstance(data, dict)
                    else data
                )

                for ev in events:
                    results.append(
                        (fecha_actual, ev)
                    )

        except Exception as e:

            print(
                f"Error conectando con la API de Euskadi "
                f"para la fecha {fecha_actual}: {e}"
            )

        fecha_actual += timedelta(days=1)

    print(
        f"Descargados {len(results)} registros totales "
        f"de la API de Euskadi.\n"
    )

    # ========================================================
    # 2. CARGAR REGISTROS EXISTENTES DE SUPABASE
    # ========================================================

    try:

        existentes_resp = (
            supabase
            .table("licitaciones")
            .select(
                "id, enlace, titulo, organo, fuente, "
                "fecha, importe, tipo_contrato, cpv, "
                "fecha_fin, es_novedad, es_actualizada"
            )
            .execute()
        )

        mapa_enlaces = {}
        registros_existentes = set()

        # IDs de Euskadi que actualmente tienen
        # algún flag activo y que debemos resetear
        ids_flags_euskadi = []

        for item in existentes_resp.data:

            # ----------------------------------------------
            # Mapa por enlace
            # ----------------------------------------------

            enlace_item = item.get("enlace")

            if enlace_item:
                mapa_enlaces[enlace_item] = item

            # ----------------------------------------------
            # Clave para detectar duplicados
            # ----------------------------------------------

            titulo_item = (
                str(item.get("titulo", ""))
                .strip()
                .lower()
            )

            organo_item = normalizar_organo(
                item.get("organo", "")
            )

            if titulo_item or organo_item:

                registros_existentes.add(
                    (
                        titulo_item,
                        organo_item
                    )
                )

            # ----------------------------------------------
            # Guardar IDs que tienen flags activos
            # ----------------------------------------------

            fuente_item = str(
                item.get("fuente", "")
            )

            if (
                "euskadi" in fuente_item.lower()
                and (
                    item.get("es_novedad") is True
                    or item.get("es_actualizada") is True
                )
            ):

                ids_flags_euskadi.append(
                    item["id"]
                )

        print(
            f"Registros cargados desde Supabase "
            f"para validación: {len(existentes_resp.data)}"
        )

    except Exception as e:

        print(
            f"Error conectando con Supabase para lectura: {e}"
        )

        return

    # ========================================================
    # 2.1. RESETEAR FLAGS ANTERIORES DE EUSKADI
    # ========================================================

    if ids_flags_euskadi:
    
        print(
            f"Reseteando flags anteriores de "
            f"{len(ids_flags_euskadi)} registros Euskadi..."
        )
    
        tamano_reset = 25
        max_intentos_reset = 3
    
        reset_correcto = True
        total_reseteadas = 0
    
        for i in range(
            0,
            len(ids_flags_euskadi),
            tamano_reset
        ):
    
            lote_ids = ids_flags_euskadi[
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
    
                    total_reseteadas += len(lote_ids)
    
                    print(
                        f"  -> Lote de flags "
                        f"{num_lote_reset} reseteado con éxito "
                        f"({len(lote_ids)} registros)."
                    )
    
                    exito_lote = True
                    break
    
                except Exception as e:
    
                    print(
                        f"  -> Intento {intento}/"
                        f"{max_intentos_reset} fallido "
                        f"para lote de flags "
                        f"{num_lote_reset}: {e}"
                    )
    
                    if intento < max_intentos_reset:
    
                        time.sleep(
                            2 * intento
                        )
    
                    else:
    
                        print(
                            f"  -> Error definitivo al resetear "
                            f"el lote de flags "
                            f"{num_lote_reset}."
                        )
    
                        reset_correcto = False
    
            if not exito_lote:
                continue
    
        if reset_correcto:
    
            print(
                f"Flags anteriores reseteados correctamente: "
                f"{total_reseteadas} registros."
            )
    
        else:
    
            print(
                "Aviso: no se pudieron resetear todos "
                "los flags anteriores."
            )
    
    else:
    
        print(
            "No hay flags anteriores de Euskadi "
            "que resetear."
        )

    # ========================================================
    # CONTADORES
    # ========================================================

    licitaciones_validas = []

    filtrados_caducados = 0
    filtrados_duplicados = 0

    registros_actualizados_count = 0
    registros_sin_cambios_count = 0

    enlaces_ya_procesados_en_sesion = set()
    claves_sesion = set()

    # ========================================================
    # 3. PROCESAR REGISTROS DE LA API
    # ========================================================

    for fecha_ev, aviso in results:

        # ----------------------------------------------------
        # DATOS BÁSICOS
        # ----------------------------------------------------

        enlace = (
            aviso.get("urlEs")
            or aviso.get("mainEntityOfPage", "")
        )

        codigo_item = (
            aviso.get("record")
            or aviso.get("id", "")
        )

        if not enlace:

            enlace = (
                "https://www.contratacion.euskadi.eus/"
                "webkpe00-kpeperfi/es/contenidos/"
                f"anuncio_contratacion/{codigo_item}/"
                "es_doc/index.html"
            )

        titulo_str = str(
            aviso.get("object")
            or aviso.get("nameEs")
            or aviso.get("nameEu")
            or "Sin título"
        ).strip()

        organo_raw = str(
            aviso.get("adjudicatorEs")
            or aviso.get("socialReason")
            or "No especificado"
        ).strip()

        importe_raw = (
            aviso.get("budgetWithoutVAT")
            or aviso.get("awardAmountWithoutVAT")
            or 0.0
        )

        importe = limpiar_importe_html(
            importe_raw
        )

        tipo_contrato = "No especificado"
        cpv = "No especificado"

        # ----------------------------------------------------
        # FECHA FIN
        # ----------------------------------------------------

        fecha_fin_str = "No especificada"

        deadline_raw = (
            aviso.get("endDate")
            or aviso.get("contractEndDate")
        )

        if deadline_raw:

            fecha_fin_str = deadline_raw[:10]

            try:

                cierre_date = datetime.strptime(
                    fecha_fin_str,
                    "%Y-%m-%d"
                ).date()

                if cierre_date < hoy_date:

                    filtrados_caducados += 1
                    continue

            except ValueError:
                pass

        # ----------------------------------------------------
        # FECHA PUBLICACIÓN
        # ----------------------------------------------------

        fecha_pub = str(
            aviso.get("startDate")
            or aviso.get("awardDate")
            or ""
        )[:10]

        # ====================================================
        # CONSULTAR DETALLE DE LA LICITACIÓN
        # ====================================================

        if codigo_item:

            try:

                url_detalle = (
                    "https://api.euskadi.eus/"
                    "procurements/contracting-notices/"
                    f"{codigo_item}"
                )

                resp_detalle = requests.get(
                    url_detalle,
                    headers={
                        "Accept": "application/json"
                    },
                    timeout=5
                )

                if resp_detalle.status_code == 200:

                    det_data = resp_detalle.json()

                    # ----------------------------------------
                    # TÍTULO
                    # ----------------------------------------

                    if det_data.get("object"):

                        titulo_str = str(
                            det_data.get("object")
                        ).strip()

                    # ----------------------------------------
                    # ÓRGANO
                    # ----------------------------------------

                    auth_name = (
                        det_data
                        .get("contractingAuthority", {})
                        .get("name")
                    )

                    org_name = (
                        det_data
                        .get("entity", {})
                        .get("org", {})
                        .get("name")
                    )

                    if auth_name:

                        organo_raw = auth_name

                    elif org_name:

                        organo_raw = org_name

                    # ----------------------------------------
                    # IMPORTE
                    # ----------------------------------------

                    if (
                        det_data.get(
                            "budgetWithoutVAT"
                        ) is not None
                    ):

                        importe = limpiar_importe_html(
                            det_data.get(
                                "budgetWithoutVAT"
                            )
                        )

                    # ----------------------------------------
                    # TIPO DE CONTRATO
                    # ----------------------------------------

                    ct_obj = det_data.get(
                        "contractType"
                    )

                    if isinstance(ct_obj, dict):

                        tipo_contrato = ct_obj.get(
                            "name",
                            "No especificado"
                        )

                    elif isinstance(ct_obj, str):

                        tipo_contrato = ct_obj

                    # ----------------------------------------
                    # CPV
                    # ----------------------------------------

                    cpv_raw = (
                        det_data.get("CPV")
                        or det_data
                        .get("contractingAuthority", {})
                        .get(
                            "CPV",
                            "No especificado"
                        )
                    )

                    if isinstance(cpv_raw, list):

                        cpv_nombres = [
                            c.get("name", "")
                            for c in cpv_raw
                            if (
                                isinstance(c, dict)
                                and c.get("name")
                            )
                        ]

                        cpv = (
                            ", ".join(cpv_nombres)
                            if cpv_nombres
                            else str(cpv_raw)
                        )

                    elif isinstance(cpv_raw, dict):

                        cpv = cpv_raw.get(
                            "name",
                            str(cpv_raw)
                        )

                    else:

                        cpv = str(cpv_raw)

            except Exception as ex:

                print(
                    f"Error consultando detalle para "
                    f"{codigo_item}: {ex}"
                )

        # ====================================================
        # REFUERZO MEDIANTE SCRAPING HTML
        # ====================================================

        if enlace:

            try:

                headers_html = {
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64)"
                    )
                }

                resp_html = requests.get(
                    enlace,
                    headers=headers_html,
                    timeout=8
                )

                if resp_html.status_code == 200:

                    soup = BeautifulSoup(
                        resp_html.text,
                        "html.parser"
                    )

                    # ----------------------------------------
                    # Buscar campos <dt> / <dd>
                    # ----------------------------------------

                    for dt in soup.find_all("dt"):

                        dt_text = (
                            dt
                            .get_text(strip=True)
                            .lower()
                        )

                        dd = dt.find_next_sibling("dd")

                        if not dd:
                            continue

                        dd_text = dd.get_text(
                            strip=True
                        )

                        # Tipo de contrato
                        if (
                            "tipo de contrato"
                            in dt_text
                            and tipo_contrato
                            == "No especificado"
                        ):

                            tipo_contrato = dd_text

                        # Importe / presupuesto
                        if (
                            (
                                "presupuesto" in dt_text
                                or "importe" in dt_text
                            )
                            and importe == 0.0
                        ):

                            importe_parsed = (
                                limpiar_importe_html(
                                    dd_text
                                )
                            )

                            if importe_parsed > 0:

                                importe = importe_parsed

                    # ----------------------------------------
                    # Respaldo para importe
                    # ----------------------------------------

                    if importe == 0.0:

                        for dd in soup.find_all("dd"):

                            txt_dd = dd.get_text(
                                strip=True
                            )

                            if re.match(
                                r'^\d{1,3}'
                                r'(\.\d{3})*'
                                r'(,\d+)?$',
                                txt_dd
                            ):

                                val = (
                                    limpiar_importe_html(
                                        txt_dd
                                    )
                                )

                                if val > 100:

                                    importe = val
                                    break

            except Exception as html_ex:

                print(
                    "Aviso: No se pudo hacer scraping "
                    f"web complementario en {enlace}: "
                    f"{html_ex}"
                )

        # ====================================================
        # NORMALIZACIÓN
        # ====================================================

        organo_str = organo_raw

        organo_base = normalizar_organo(
            organo_raw
        )

        lugar_ejecucion = procesar_lugar_euskadi(
            "País Vasco"
        )

        clave_duplicado = (
            titulo_str.lower(),
            organo_base
        )

        # ====================================================
        # EVITAR DUPLICADOS DENTRO DE LA PROPIA SESIÓN
        # ====================================================

        if (
            enlace in enlaces_ya_procesados_en_sesion
            or clave_duplicado in claves_sesion
        ):

            filtrados_duplicados += 1
            continue

        enlaces_ya_procesados_en_sesion.add(
            enlace
        )

        claves_sesion.add(
            clave_duplicado
        )

        # ====================================================
        # BUSCAR REGISTRO EXISTENTE
        # ====================================================

        registro_existente = mapa_enlaces.get(
            enlace
        )

        if (
            not registro_existente
            and clave_duplicado in registros_existentes
        ):

            for item_b in mapa_enlaces.values():

                t_b = (
                    str(item_b.get("titulo", ""))
                    .strip()
                    .lower()
                )

                o_b = normalizar_organo(
                    item_b.get("organo", "")
                )

                if (
                    t_b,
                    o_b
                ) == clave_duplicado:

                    registro_existente = item_b
                    break

        # ====================================================
        # SI YA EXISTE
        # ====================================================

        if registro_existente:

            fuente_actual = str(
                registro_existente.get(
                    "fuente",
                    ""
                )
            )

            if "euskadi" not in fuente_actual.lower():

                fuente_final = (
                    f"{fuente_actual}, Euskadi"
                    if fuente_actual
                    else "Euskadi"
                )

            else:

                fuente_final = fuente_actual

            # ----------------------------------------------
            # Comprobar si ha habido cambios
            # ----------------------------------------------

            cambios = False

            # Título
            if str(
                registro_existente.get(
                    "titulo",
                    ""
                )
            ).strip() != titulo_str.strip():

                cambios = True

            # Órgano
            if str(
                registro_existente.get(
                    "organo",
                    ""
                )
            ).strip() != organo_str.strip():

                cambios = True

            # Fecha
            if str(
                registro_existente.get(
                    "fecha",
                    ""
                )
            ).strip() != fecha_pub.strip():

                cambios = True

            # Importe
            try:

                importe_existente = float(
                    registro_existente.get(
                        "importe"
                    ) or 0
                )

                if abs(
                    importe_existente - importe
                ) > 0.01:

                    cambios = True

            except (
                ValueError,
                TypeError
            ):

                cambios = True

            # Tipo de contrato
            if str(
                registro_existente.get(
                    "tipo_contrato",
                    ""
                )
            ).strip() != tipo_contrato.strip():

                cambios = True

            # CPV
            if str(
                registro_existente.get(
                    "cpv",
                    ""
                )
            ).strip() != cpv.strip():

                cambios = True

            # Fecha fin
            if str(
                registro_existente.get(
                    "fecha_fin",
                    ""
                )
            ).strip() != fecha_fin_str.strip():

                cambios = True

            # ----------------------------------------------
            # Preparar actualización
            # ----------------------------------------------

            datos_actualizar = {
                "fuente": fuente_final,
                "es_novedad": False,
                "es_actualizada": cambios
            }

            # ----------------------------------------------
            # Si ha cambiado, actualizar todos los datos
            # ----------------------------------------------

            if cambios:

                texto_completo = (
                    f"passage: Título: {titulo_str}. "
                    f"Órgano: {organo_str}. "
                    f"CPV: {cpv}. "
                    f"Tipo de contrato: "
                    f"{tipo_contrato}. "
                    f"Lugar: {lugar_ejecucion}. "
                    f"Importe: {importe} EUR."
                )

                embedding = encoder.encode(
                    texto_completo
                ).tolist()

                datos_actualizar.update({

                    "titulo": titulo_str,

                    "organo": organo_str,

                    "fecha": fecha_pub,

                    "importe": importe,

                    "tipo_contrato": tipo_contrato,

                    "cpv": cpv,

                    "fecha_fin": fecha_fin_str,

                    "lugar_ejecucion": lugar_ejecucion,

                    "texto_completo": texto_completo,

                    "embedding": embedding
                })

            # ----------------------------------------------
            # Actualizar registro individualmente
            # ----------------------------------------------

            try:

                (
                    supabase
                    .table("licitaciones")
                    .update(datos_actualizar)
                    .eq(
                        "id",
                        registro_existente["id"]
                    )
                    .execute()
                )

                if cambios:

                    registros_actualizados_count += 1

                    print(
                        "  -> Licitación actualizada: "
                        f"{titulo_str[:80]}"
                    )

                else:

                    registros_sin_cambios_count += 1

            except Exception as e:

                print(
                    "Error actualizando registro existente "
                    f"{registro_existente.get('id')}: {e}"
                )

            continue

        # ====================================================
        # SI NO EXISTE -> NUEVA LICITACIÓN
        # ====================================================

        fuente_final = "Euskadi"

        texto_completo = (
            f"passage: Título: {titulo_str}. "
            f"Órgano: {organo_str}. "
            f"CPV: {cpv}. "
            f"Tipo de contrato: {tipo_contrato}. "
            f"Lugar: {lugar_ejecucion}. "
            f"Importe: {importe} EUR."
        )

        embedding = encoder.encode(
            texto_completo
        ).tolist()

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
            "fuente": fuente_final
        }

        licitaciones_validas.append(
            elemento
        )

    # ========================================================
    # 4. LIMPIEZA AUTOMÁTICA DE CADUCADAS
    # ========================================================

    try:

        todos_db = (
            supabase
            .table("licitaciones")
            .select(
                "id, enlace, fecha_fin"
            )
            .ilike(
                "fuente",
                "%Euskadi%"
            )
            .execute()
        )

        ids_a_borrar = []

        for item in todos_db.data:

            f_fin = item.get(
                "fecha_fin"
            )

            if (
                f_fin
                and f_fin != "No especificada"
            ):

                try:

                    f_cierre = datetime.strptime(
                        f_fin,
                        "%Y-%m-%d"
                    ).date()

                    if f_cierre < hoy_date:

                        ids_a_borrar.append(
                            item["id"]
                        )

                except ValueError:
                    pass

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
                "licitaciones caducadas de Supabase."
            )

    except Exception as e:

        print(
            "Error en la limpieza de caducadas: "
            f"{e}"
        )

    # ========================================================
    # 5. ESTADÍSTICAS
    # ========================================================

    print(
        "\n--- ESTADÍSTICAS EUSKADI ---"
    )

    print(
        "Descartados por fecha caducada: "
        f"{filtrados_caducados}"
    )

    print(
        "Duplicados evitados durante la sesión: "
        f"{filtrados_duplicados}"
    )

    print(
        "Registros existentes sin cambios: "
        f"{registros_sin_cambios_count}"
    )

    print(
        "Registros existentes actualizados: "
        f"{registros_actualizados_count}"
    )

    print(
        "Nuevas licitaciones válidas listas "
        f"para insertar: {len(licitaciones_validas)}"
    )

    # ========================================================
    # 6. INSERTAR NUEVAS LICITACIONES POR LOTES
    # ========================================================

    if licitaciones_validas:

        print(
            "\nSubiendo nuevas licitaciones de Euskadi "
            "a Supabase..."
        )

        tamano_lote = 15
        max_intentos = 3

        subidas_exitosas = 0

        total_a_subir = len(
            licitaciones_validas
        )

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
                        .upsert(
                            lote,
                            on_conflict="enlace"
                        )
                        .execute()
                    )

                    subidas_exitosas += len(
                        lote
                    )

                    print(
                        f"  -> Lote Euskadi "
                        f"{num_lote} procesado con éxito "
                        f"({len(lote)} registros)."
                    )

                    exito = True
                    break

                except Exception as e:

                    print(
                        f"Intento {intento}/"
                        f"{max_intentos} fallido "
                        f"para lote Euskadi "
                        f"{num_lote}: {e}"
                    )

                    if intento < max_intentos:

                        time.sleep(
                            2 * intento
                        )

                    else:

                        print(
                            f"Error definitivo al subir "
                            f"lote Euskadi {num_lote}."
                        )

        print(
            "\n¡Sincronización de Euskadi "
            f"completada con éxito! "
            f"Se han subido {subidas_exitosas} "
            f"de {total_a_subir} "
            "licitaciones nuevas."
        )

    else:

        print(
            "\nNo hay nuevas licitaciones de Euskadi "
            "para insertar."
        )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    sincronizar_licitaciones_euskadi()
