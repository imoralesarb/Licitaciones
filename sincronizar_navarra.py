from datetime import datetime, date, timedelta
import os
import time
import re
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import requests

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small)...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def limpiar_cpv(cpv_raw):
    """Extrae los primeros 8 dígitos del CPV."""
    if not cpv_raw or cpv_raw == "No especificado":
        return "No especificado"

    cpv_str = str(cpv_raw).strip()
    match = re.search(r"(\d{8}(?:-\d)?)", cpv_str)

    if match:
        return match.group(1)

    return cpv_str[:20]


def procesar_lugar_navarra(lugar_raw):
    """Añade Navarra al lugar de ejecución si no lo especifica."""
    lugar_limpio = str(lugar_raw).strip() if lugar_raw else "No especificado"

    if lugar_limpio == "No especificado" or not lugar_limpio:
        return "Navarra"

    if "navarra" not in lugar_limpio.lower():
        return f"{lugar_limpio}, Navarra"

    return lugar_limpio


def normalizar_organo(texto):
    """Normaliza el órgano para detectar duplicados."""
    if not texto:
        return ""

    texto = str(texto).lower().strip()

    # Eliminar acentos
    texto = re.sub(r"[áàäâ]", "a", texto)
    texto = re.sub(r"[éèëê]", "e", texto)
    texto = re.sub(r"[íìïî]", "i", texto)
    texto = re.sub(r"[óòöô]", "o", texto)
    texto = re.sub(r"[úùüû]", "u", texto)

    # Eliminar caracteres especiales
    texto = re.sub(r"[^a-z0-9\s]", "", texto)

    # Normalizar espacios
    return re.sub(r"\s+", " ", texto)


def normalizar_fuentes(fuente):
    """Convierte la cadena de fuentes en una lista limpia."""
    if not fuente:
        return []

    return [
        f.strip()
        for f in str(fuente).split(",")
        if f.strip()
    ]


def contiene_fuente(fuente_actual, nombre_fuente):
    """Comprueba si una fuente concreta está presente."""
    fuentes = normalizar_fuentes(fuente_actual)

    return any(
        f.casefold() == nombre_fuente.casefold()
        for f in fuentes
    )


def añadir_fuente(fuente_actual, nombre_fuente):
    """Añade una fuente sin duplicarla."""
    fuentes = normalizar_fuentes(fuente_actual)

    if not any(
        f.casefold() == nombre_fuente.casefold()
        for f in fuentes
    ):
        fuentes.append(nombre_fuente)

    return ", ".join(fuentes)


def quitar_fuente(fuente_actual, nombre_fuente):
    """Elimina únicamente una fuente concreta."""
    fuentes = normalizar_fuentes(fuente_actual)

    fuentes = [
        f for f in fuentes
        if f.casefold() != nombre_fuente.casefold()
    ]

    return ", ".join(fuentes)


# ============================================================
# SINCRONIZACIÓN NAVARRA
# ============================================================

def sincronizar_licitaciones_navarra():

    hoy_date = datetime.now().date()
    limite_fecha = hoy_date - timedelta(days=2)

    print(
        f"Filtrando licitaciones publicadas desde "
        f"{limite_fecha} hasta {hoy_date}"
    )

    resource_id = "dda1af7c-0dcd-4992-9852-ded6b1e7625d"

    url_api = (
        "https://datosabiertos.navarra.es/es/api/3/action/"
        f"datastore_search?resource_id={resource_id}&limit=1000"
    )

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    FUENTE_NAVARRA = "Contratación Navarra"

    # ========================================================
    # 1. DESCARGAR DATOS DE NAVARRA
    # ========================================================

    print("Consultando la API de Contratación de Navarra...")

    try:
        response = requests.get(
            url_api,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:
            print(f"Error HTTP {response.status_code}")
            return

        data = response.json()

        results = data.get(
            "result", {}
        ).get(
            "records", []
        )

    except Exception as e:
        print(
            f"Error conectando con la API de Navarra: {e}"
        )
        return

    print(
        f"Total registros obtenidos de la API Navarra: "
        f"{len(results)}"
    )

    # ========================================================
    # 2. CARGAR REGISTROS EXISTENTES DE SUPABASE
    # ========================================================

    try:
        existentes_resp = (
            supabase.table("licitaciones")
            .select(
                "id, enlace, titulo, organo, fuente, fecha, "
                "importe, tipo_contrato, cpv, fecha_fin, "
                "es_novedad, es_actualizada"
            )
            .execute()
        )

        registros_db = {}
        registros_existentes = set()
        ids_flags_navarra = []

        for item in existentes_resp.data:

            # ----------------------------------------------
            # Mapa por enlace
            # ----------------------------------------------

            enlace_item = item.get("enlace")

            if enlace_item:
                registros_db[enlace_item] = item

            # ----------------------------------------------
            # Mapa por título + órgano
            # ----------------------------------------------

            titulo_item = str(
                item.get("titulo") or ""
            ).strip().lower()

            organo_item = normalizar_organo(
                item.get("organo", "")
            )

            if titulo_item or organo_item:
                registros_existentes.add(
                    (titulo_item, organo_item)
                )

            # ----------------------------------------------
            # Registros de Navarra con etiquetas anteriores
            # ----------------------------------------------

            fuente_item = str(
                item.get("fuente", "")
            )

            if contiene_fuente(
                fuente_item,
                FUENTE_NAVARRA
            ) and (
                item.get("es_novedad") is True
                or item.get("es_actualizada") is True
            ):
                ids_flags_navarra.append(
                    item["id"]
                )

        print(
            "Registros cargados desde Supabase para "
            f"validación: {len(existentes_resp.data)}"
        )

    except Exception as e:

        print(
            f"Error conectando con Supabase para lectura: {e}"
        )
        return

    # ========================================================
    # 2.1. RESETEAR ETIQUETAS ANTERIORES DE NAVARRA
    # ========================================================

    if ids_flags_navarra:

        print(
            f"Reseteando etiquetas anteriores de "
            f"{len(ids_flags_navarra)} registros de Navarra..."
        )

        tamano_reset = 25
        max_intentos_reset = 3

        reset_correcto = True
        total_reseteadas = 0

        for i in range(
            0,
            len(ids_flags_navarra),
            tamano_reset
        ):

            lote_ids = ids_flags_navarra[
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
                        time.sleep(2 * intento)

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
            "No hay etiquetas anteriores de Navarra "
            "que resetear."
        )

    # ========================================================
    # 3. PROCESAR LICITACIONES
    # ========================================================

    licitaciones_validas = []

    enlaces_procesados_sesion = set()

    claves_procesadas_sesion = set()

    for i, aviso in enumerate(
        results,
        1
    ):

        # ----------------------------------------------
        # FECHA PUBLICACIÓN
        # ----------------------------------------------

        fecha_pub_str = aviso.get(
            "FechaPublicacion",
            ""
        )

        if not fecha_pub_str:
            continue

        try:

            dt_pub = datetime.strptime(
                fecha_pub_str[:10],
                "%d/%m/%Y"
            )

            pub_date = dt_pub.date()

            if (
                pub_date < limite_fecha
                or pub_date > hoy_date
            ):
                continue

            fecha_pub = dt_pub.strftime(
                "%Y-%m-%d"
            )

        except ValueError:

            continue

        # ----------------------------------------------
        # ENLACE
        # ----------------------------------------------

        ckan_id = aviso.get(
            "_id",
            i
        )

        enlace = (
            "https://hacienda.navarra.es/"
            "sicpportal/mtoBuscadorAnuncios.aspx#"
            f"{ckan_id}"
        )

        if (
            not enlace
            or enlace in enlaces_procesados_sesion
        ):
            continue

        enlaces_procesados_sesion.add(
            enlace
        )

        # ----------------------------------------------
        # TÍTULO Y ÓRGANO
        # ----------------------------------------------

        titulo = str(
            aviso.get("BreveDescripcion")
            or "Sin descripción"
        ).strip()

        organo = str(
            aviso.get("Organo")
            or "No especificado"
        ).strip()

        entidad = str(
            aviso.get("Entidad")
            or "No especificada"
        ).strip()

        organo_completo = (
            f"{entidad} - {organo}"
            if entidad
            else organo
        )

        titulo_normalizado = (
            titulo.lower().strip()
        )

        organo_normalizado = normalizar_organo(
            organo_completo
        )

        clave_duplicado = (
            titulo_normalizado,
            organo_normalizado
        )

        # ----------------------------------------------
        # IMPORTE
        # ----------------------------------------------

        importe_val = (
            aviso.get("PrecioLicitacion")
            or aviso.get("ValorEstimado")
        )

        try:

            val_limpio = (
                str(importe_val)
                .replace("€", "")
                .replace("EUR", "")
                .replace(".", "")
                .replace(",", ".")
                .strip()
            )

            importe = (
                float(val_limpio)
                if val_limpio
                else 0.0
            )

        except (
            ValueError,
            TypeError
        ):

            importe = 0.0

        # ----------------------------------------------
        # CPV
        # ----------------------------------------------

        cpv = limpiar_cpv(
            aviso.get(
                "CPV",
                "No especificado"
            )
        )

        # ----------------------------------------------
        # LUGAR
        # ----------------------------------------------

        lugar_bruto = aviso.get(
            "LugarEjecucion",
            "Navarra"
        )

        lugar_ejecucion = procesar_lugar_navarra(
            lugar_bruto
        )

        # ----------------------------------------------
        # TIPO DE CONTRATO
        # ----------------------------------------------

        tipo_contrato = str(
            aviso.get(
                "TipoContrato",
                "No especificado"
            )
        ).capitalize()

        # ----------------------------------------------
        # FECHA FIN
        # ----------------------------------------------

        fecha_fin_str = "No especificada"

        # ----------------------------------------------
        # TEXTO PARA EMBEDDING
        # ----------------------------------------------

        texto_completo = (
            f"passage: Título: {titulo}. "
            f"Órgano: {organo_completo}. "
            f"Tipo Contrato: {tipo_contrato}. "
            f"Lugar: {lugar_ejecucion}. "
            f"Importe: {importe} EUR. "
            f"CPV: {cpv}."
        )

        # ==================================================
        # 3.1. EXISTE POR ENLACE
        # ==================================================

        if enlace in registros_db:

            reg_antiguo = registros_db[
                enlace
            ]

            fuente_actual = str(
                reg_antiguo.get("fuente") or ""
            )

            tipo_actual = (
                reg_antiguo.get(
                    "tipo_contrato"
                )
            )

            actualizar_datos = {}

            # ------------------------------------------
            # Añadir Navarra como fuente
            # ------------------------------------------

            if not contiene_fuente(
                fuente_actual,
                FUENTE_NAVARRA
            ):

                actualizar_datos["fuente"] = añadir_fuente(
                    fuente_actual,
                    FUENTE_NAVARRA
                )

            # ------------------------------------------
            # Completar tipo de contrato
            # ------------------------------------------

            if (
                not tipo_actual
                or tipo_actual == "No especificado"
            ) and (
                tipo_contrato != "No especificado"
            ):

                actualizar_datos[
                    "tipo_contrato"
                ] = tipo_contrato

            # ------------------------------------------
            # Detectar cambios
            # ------------------------------------------

            es_actualizado = (
                reg_antiguo.get("titulo")
                != titulo
                or
                reg_antiguo.get("importe")
                != importe
            )

            if es_actualizado:

                actualizar_datos[
                    "es_actualizada"
                ] = True

            # ------------------------------------------
            # Actualizar
            # ------------------------------------------

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
                        f"Error actualizando registro "
                        f"existente {enlace}: {e}"
                    )

            continue

        # ==================================================
        # 3.2. EXISTE POR TÍTULO + ÓRGANO
        # ==================================================

        if clave_duplicado in registros_existentes:

            # Buscar el registro correspondiente
            # para añadir Navarra como fuente

            registro_duplicado = None

            for reg in registros_db.values():

                titulo_db = str(
                    reg.get("titulo") or ""
                ).strip().lower()

                organo_db = normalizar_organo(
                    reg.get("organo", "")
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
                    FUENTE_NAVARRA
                ):

                    nueva_fuente = añadir_fuente(
                        fuente_actual,
                        FUENTE_NAVARRA
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
                                registro_duplicado["id"]
                            )
                            .execute()
                        )

                        registro_duplicado[
                            "fuente"
                        ] = nueva_fuente

                    except Exception as e:

                        print(
                            f"Error añadiendo fuente "
                            f"Navarra al duplicado: {e}"
                        )

            continue

        # ==================================================
        # 3.3. NUEVA LICITACIÓN
        # ==================================================

        # Evitar duplicados dentro de la misma ejecución

        if clave_duplicado in claves_procesadas_sesion:
            continue

        claves_procesadas_sesion.add(
            clave_duplicado
        )

        embedding = encoder.encode(
            texto_completo
        ).tolist()

        elemento = {
            "titulo": titulo,
            "organo": organo_completo,
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
            "fuente": FUENTE_NAVARRA,
        }

        licitaciones_validas.append(
            elemento
        )

        # Importante:
        # añadimos también la clave a los registros
        # existentes para evitar otra inserción en
        # la misma ejecución.
        registros_existentes.add(
            clave_duplicado
        )

    # ========================================================
    # 4. LIMPIEZA AUTOMÁTICA DE CADUCADAS
    # ========================================================

    try:

        todos_db = (
            supabase
            .table("licitaciones")
            .select(
                "id, enlace, fecha_fin, fuente"
            )
            .ilike(
                "fuente",
                "%Contratación Navarra%"
            )
            .execute()
        )

        ids_a_borrar = []
        ids_a_actualizar = []

        for item in todos_db.data:

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
                    f_fin,
                    "%Y-%m-%d"
                ).date()

            except ValueError:

                continue

            if f_cierre < hoy_date:

                fuente_actual = str(
                    item.get("fuente") or ""
                )

                fuentes = normalizar_fuentes(
                    fuente_actual
                )

                # --------------------------------------
                # Navarra es la única fuente
                # --------------------------------------

                if (
                    len(fuentes) == 1
                    and contiene_fuente(
                        fuente_actual,
                        FUENTE_NAVARRA
                    )
                ):

                    ids_a_borrar.append(
                        item["id"]
                    )

                # --------------------------------------
                # Hay más fuentes
                # --------------------------------------

                elif contiene_fuente(
                    fuente_actual,
                    FUENTE_NAVARRA
                ):

                    nueva_fuente = quitar_fuente(
                        fuente_actual,
                        FUENTE_NAVARRA
                    )

                    ids_a_actualizar.append(
                        (
                            item["id"],
                            nueva_fuente
                        )
                    )

        # ----------------------------------------------
        # Borrar registros cuya única fuente es Navarra
        # ----------------------------------------------

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
                f"fuente era Navarra."
            )

        # ----------------------------------------------
        # Quitar solo Navarra en fuentes combinadas
        # ----------------------------------------------

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
                    f"Error quitando fuente Navarra "
                    f"del registro {registro_id}: {e}"
                )

        if ids_a_actualizar:

            print(
                f"Quitada la fuente Navarra de "
                f"{len(ids_a_actualizar)} licitaciones "
                f"caducadas que tenían otras fuentes."
            )

    except Exception as e:

        print(
            f"Error en la limpieza de caducadas: {e}"
        )

    # ========================================================
    # 5. INSERTAR EN SUPABASE
    # ========================================================

    if licitaciones_validas:

        total_a_subir = len(
            licitaciones_validas
        )

        print(
            f"Subiendo un total de {total_a_subir} "
            f"licitaciones publicadas entre "
            f"{limite_fecha} y {hoy_date} a Supabase..."
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

                    # INSERT en lugar de UPSERT.
                    # Los duplicados ya se controlan arriba.

                    (
                        supabase
                        .table("licitaciones")
                        .insert(lote)
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
                        f"⚠️ Intento "
                        f"{intento}/{max_intentos} "
                        f"fallido para lote Navarra "
                        f"{num_lote}: {e}"
                    )

                    if intento < max_intentos:
                        time.sleep(2 * intento)

                    else:

                        print(
                            f"❌ Error definitivo al "
                            f"subir lote Navarra "
                            f"{num_lote}."
                        )

            if not exito:
                continue

        print(
            f"Sincronización completada con éxito. "
            f"Se han insertado {subidas_exitosas} "
            f"de {total_a_subir} licitaciones nuevas."
        )

    else:

        print(
            f"No hay licitaciones publicadas entre "
            f"{limite_fecha} y {hoy_date} para procesar."
        )


if __name__ == "__main__":
    sincronizar_licitaciones_navarra()
