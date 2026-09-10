from datetime import datetime, date
import os
import requests
import lxml.etree as ET
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client
from urllib3.util.retry import Retry
import time

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small) para auditoría...")
encoder = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu"
)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cac": "urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2",
    "cac-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2",
    "cbc-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2",
}

ESTADOS_CERRADOS = [
    "EV",
    "ADJ",
    "RES",
    "ANUL",
    "FOR",
    "AS",
    "RE",
    "CAN"
]

TAMANO_LOTE = 10


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def crear_sesion_robusta():
    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )

    adapter = HTTPAdapter(
        max_retries=retries
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def normalizar_fuentes(fuente):
    """
    Separa correctamente las fuentes cuando están combinadas.

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
    Comprueba si una fuente concreta está presente,
    incluso cuando existen varias fuentes combinadas.
    """

    fuentes = normalizar_fuentes(
        fuente_actual
    )

    return any(
        f.casefold() == nombre_fuente.casefold()
        for f in fuentes
    )


def quitar_fuente(fuente, nombre_fuente):
    """
    Elimina únicamente una fuente concreta de una
    lista de fuentes combinadas.

    Ejemplo:
        "TED, Licitaciones Generales PLACSP"

    quitando:
        "Licitaciones Generales PLACSP"

    devuelve:
        "TED"
    """

    fuentes = normalizar_fuentes(
        fuente
    )

    fuentes_restantes = [
        f
        for f in fuentes
        if f.casefold() != nombre_fuente.casefold()
    ]

    return ", ".join(
        fuentes_restantes
    )


# ============================================================
# 3. AUDITORÍA
# ============================================================

def auditar_licitaciones_abiertas():

    hoy = date.today()

    sesion = crear_sesion_robusta()

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    total_eliminadas = 0
    total_fuentes_quitadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    total_errores = 0

    lote_contador = 1

    ids_procesados = []

    print(
        "🔍 Iniciando auditoría por lotes de "
        "licitaciones PLACSP con fecha de cierre "
        "'No especificada'...\n"
    )

    # ========================================================
    # BUCLE PRINCIPAL
    # ========================================================

    while True:

        try:

            # ------------------------------------------------
            # Buscar registros con fecha fin desconocida
            # ------------------------------------------------

            query = (
                supabase
                .table("licitaciones")
                .select(
                    "id,enlace,titulo,organo,fuente,"
                    "tipo_contrato,cpv,lugar_ejecucion,"
                    "importe,fecha_fin,texto_completo"
                )
                .eq(
                    "fecha_fin",
                    "No especificada"
                )
                .limit(
                    TAMANO_LOTE
                )
            )

            # ------------------------------------------------
            # No volver a procesar IDs de esta ejecución
            # ------------------------------------------------

            if ids_procesados:

                query = query.not_.in_(
                    "id",
                    ids_procesados
                )

            response = query.execute()

            registros = response.data or []

        except Exception as e:

            print(
                f"❌ Error al consultar Supabase: {e}"
            )

            total_errores += 1

            break

        # ====================================================
        # FIN
        # ====================================================

        if not registros:

            print(
                "\n🎉 ¡Proceso finalizado! "
                "No quedan más licitaciones pendientes "
                "de auditar en este grupo."
            )

            break

        print(
            f"\n📦 --- Procesando Lote "
            f"{lote_contador} "
            f"({len(registros)} registros) ---"
        )

        ids_a_borrar = []

        # ====================================================
        # PROCESAR CADA REGISTRO
        # ====================================================

        for reg in registros:

            rec_id = reg.get("id")

            ids_procesados.append(
                rec_id
            )

            enlace = reg.get(
                "enlace"
            )

            titulo = reg.get(
                "titulo",
                "Sin título"
            )

            fuente = reg.get(
                "fuente",
                ""
            )

            # ------------------------------------------------
            # Identificar qué fuente PLACSP tiene el registro
            # ------------------------------------------------

            fuente_placsp = None

            if contiene_fuente(
                fuente,
                "Licitaciones Generales PLACSP"
            ):

                fuente_placsp = (
                    "Licitaciones Generales PLACSP"
                )

            elif contiene_fuente(
                fuente,
                "Licitaciones Agregadas PLACSP"
            ):

                fuente_placsp = (
                    "Licitaciones Agregadas PLACSP"
                )

            # ------------------------------------------------
            # Solo auditar registros pertenecientes a PLACSP
            # ------------------------------------------------

            if fuente_placsp is None:

                total_sin_cambios += 1

                continue

            # ------------------------------------------------
            # Sin enlace no podemos consultar el expediente
            # ------------------------------------------------

            if not enlace:

                continue

            # ------------------------------------------------
            # TED se excluye
            # ------------------------------------------------

            if "ted.europa.eu" in enlace:

                continue

            try:

                # =================================================
                # DESCARGAR EXPEDIENTE
                # =================================================

                resp = sesion.get(
                    enlace,
                    headers=headers,
                    timeout=12
                )

                if resp.status_code != 200:

                    print(
                        f"   ⚠️ Enlace no disponible "
                        f"(HTTP {resp.status_code})"
                    )

                    total_errores += 1

                    continue

                parser = ET.XMLParser(
                    recover=True
                )

                root = ET.fromstring(
                    resp.content,
                    parser=parser
                )

                # =================================================
                # 1. ESTADO ACTUAL
                # =================================================

                codigo_estado = "PUB"

                estado_el = root.find(
                    ".//cbc-place-ext:"
                    "ContractFolderStatusCode",
                    NS
                )

                if estado_el is None:

                    estado_el = root.find(
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

                # =================================================
                # Si está cerrado
                # =================================================

                if codigo_estado in ESTADOS_CERRADOS:

                    # ------------------------------------------------
                    # Comprobar si existen otras fuentes
                    # ------------------------------------------------

                    nueva_fuente = quitar_fuente(
                        fuente,
                        fuente_placsp
                    )

                    # ------------------------------------------------
                    # Si no quedan otras fuentes -> borrar registro
                    # ------------------------------------------------

                    if not nueva_fuente:

                        ids_a_borrar.append(
                            rec_id
                        )

                        print(
                            f"   🗑️ [A BORRAR - Estado "
                            f"{codigo_estado}]: "
                            f"{titulo[:50]}..."
                        )

                        total_eliminadas += 1

                    # ------------------------------------------------
                    # Si quedan otras fuentes -> quitar solo PLACSP
                    # ------------------------------------------------

                    else:

                        try:

                            supabase.table(
                                "licitaciones"
                            ).update(
                                {
                                    "fuente": nueva_fuente
                                }
                            ).eq(
                                "id",
                                rec_id
                            ).execute()

                            print(
                                f"   🔄 [FUENTE QUITADA - "
                                f"Estado {codigo_estado}]: "
                                f"{titulo[:50]}... "
                                f"-> {nueva_fuente}"
                            )

                            total_fuentes_quitadas += 1

                        except Exception as e:

                            print(
                                f"   ❌ Error quitando fuente "
                                f"de {titulo[:50]}...: {e}"
                            )

                            total_errores += 1

                    continue

                # =================================================
                # 2. BUSCAR FECHA FIN
                # =================================================

                end_date_el = root.find(
                    ".//cac:TenderingProcess/"
                    "cac:TenderSubmissionDeadlinePeriod/"
                    "cbc:EndDate",
                    NS
                )

                nueva_fecha_fin = (
                    "No especificada"
                )

                if (
                    end_date_el is not None
                    and end_date_el.text
                ):

                    nueva_fecha_fin = (
                        end_date_el.text
                        .strip()[:10]
                    )

                # =================================================
                # 3. COMPROBAR CADUCIDAD
                # =================================================

                if (
                    nueva_fecha_fin
                    != "No especificada"
                ):

                    try:

                        f_fin_date = datetime.strptime(
                            nueva_fecha_fin,
                            "%Y-%m-%d"
                        ).date()

                        if f_fin_date < hoy:

                            # ------------------------------------------------
                            # Comprobar si existen otras fuentes
                            # ------------------------------------------------

                            nueva_fuente = quitar_fuente(
                                fuente,
                                fuente_placsp
                            )

                            # ------------------------------------------------
                            # Si no quedan otras fuentes -> borrar
                            # ------------------------------------------------

                            if not nueva_fuente:

                                ids_a_borrar.append(
                                    rec_id
                                )

                                print(
                                    f"   🗑️ [A BORRAR - "
                                    f"Caducada]: "
                                    f"{titulo[:50]}..."
                                )

                                total_eliminadas += 1

                            # ------------------------------------------------
                            # Si quedan otras fuentes -> quitar PLACSP
                            # ------------------------------------------------

                            else:

                                try:

                                    supabase.table(
                                        "licitaciones"
                                    ).update(
                                        {
                                            "fuente": nueva_fuente
                                        }
                                    ).eq(
                                        "id",
                                        rec_id
                                    ).execute()

                                    print(
                                        f"   🔄 [FUENTE QUITADA - "
                                        f"Caducada]: "
                                        f"{titulo[:50]}... "
                                        f"-> {nueva_fuente}"
                                    )

                                    total_fuentes_quitadas += 1

                                except Exception as e:

                                    print(
                                        f"   ❌ Error quitando fuente "
                                        f"de {titulo[:50]}...: {e}"
                                    )

                                    total_errores += 1

                            continue

                    except ValueError:

                        pass

                # =================================================
                # 4. COMPROBAR CAMBIO DE FECHA
                # =================================================

                fecha_fin_antigua = reg.get(
                    "fecha_fin"
                )

                if (
                    nueva_fecha_fin
                    != fecha_fin_antigua
                ):

                    # ------------------------------------------------
                    # Solo actualizamos la fecha y la flag.
                    #
                    # NO recalculamos embedding.
                    # ------------------------------------------------

                    datos_actualizar = {
                        "fecha_fin": nueva_fecha_fin,
                        "es_actualizada": True
                    }

                    try:

                        supabase.table(
                            "licitaciones"
                        ).update(
                            datos_actualizar
                        ).eq(
                            "id",
                            rec_id
                        ).execute()

                        print(
                            f"   🔄 [ACTUALIZADA "
                            f"Fecha Fin: "
                            f"{nueva_fecha_fin}]: "
                            f"{titulo[:50]}..."
                        )

                        total_actualizadas += 1

                    except Exception as e:

                        print(
                            f"   ❌ Error actualizando "
                            f"{titulo[:50]}...: {e}"
                        )

                        total_errores += 1

                else:

                    total_sin_cambios += 1

                time.sleep(
                    0.2
                )

            except Exception as e:

                print(
                    f"   ⚠️ Error procesando "
                    f"enlace: {e}"
                )

                total_errores += 1

                continue

        # ========================================================
        # ELIMINACIONES DEL LOTE
        # ========================================================

        if ids_a_borrar:

            try:

                supabase.table(
                    "licitaciones"
                ).delete().in_(
                    "id",
                    ids_a_borrar
                ).execute()

                print(
                    f"   🗑️ -> ¡"
                    f"{len(ids_a_borrar)} "
                    f"licitaciones eliminadas "
                    f"de Supabase en este lote!"
                )

            except Exception as e:

                print(
                    f"   ❌ Error al eliminar "
                    f"lote en Supabase: {e}"
                )

                total_errores += 1

        lote_contador += 1

        time.sleep(
            0.5
        )

    # ============================================================
    # RESUMEN
    # ============================================================

    print(
        "\n" + "=" * 50
    )

    print(
        "📊 RESUMEN FINAL DE LA AUDITORÍA:"
    )

    print(
        f"  - Eliminadas "
        f"(cerradas/caducadas): "
        f"{total_eliminadas}"
    )

    print(
        f"  - Fuentes PLACSP quitadas: "
        f"{total_fuentes_quitadas}"
    )

    print(
        f"  - Actualizadas "
        f"(con nueva fecha): "
        f"{total_actualizadas}"
    )

    print(
        f"  - Sin cambios: "
        f"{total_sin_cambios}"
    )

    print(
        f"  - Errores: "
        f"{total_errores}"
    )

    print(
        "✅ ¡Auditoría de licitaciones "
        "completada con éxito!"
    )


# ============================================================
# 4. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    auditar_licitaciones_abiertas()
