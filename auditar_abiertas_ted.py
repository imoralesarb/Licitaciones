from datetime import datetime, date
import os
import requests
from supabase import Client, create_client
import time

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MAPEO_NUTS_TED = {
    "ES111": "A Coruña", "ES112": "Lugo", "ES113": "Ourense", "ES114": "Pontevedra",
    "ES120": "Asturias", "ES130": "Cantabria",
    "ES211": "Álava/Araba", "ES212": "Gipuzkoa", "ES213": "Bizkaia",
    "ES220": "La Rioja", "ES230": "Navarra",
    "ES241": "Huesca", "ES242": "Teruel", "ES243": "Zaragoza",
    "ES300": "Madrid",
    "ES411": "Ávila", "ES412": "Burgos", "ES413": "León", "ES414": "Palencia",
    "ES415": "Salamanca", "ES416": "Segovia", "ES417": "Soria",
    "ES418": "Valladolid", "ES419": "Zamora",
    "ES421": "Albacete", "ES422": "Ciudad Real", "ES423": "Cuenca",
    "ES424": "Guadalajara", "ES425": "Toledo",
    "ES431": "Badajoz", "ES432": "Cáceres",
    "ES511": "Barcelona", "ES512": "Girona", "ES513": "Lleida", "ES514": "Tarragona",
    "ES521": "Alicante/Alacant", "ES522": "Castellón/Castelló",
    "ES523": "Valencia/València",
    "ES531": "Eivissa y Formentera", "ES532": "Mallorca", "ES533": "Menorca",
    "ES611": "Almería", "ES612": "Cádiz", "ES613": "Córdoba",
    "ES614": "Granada", "ES615": "Huelva", "ES616": "Jaén",
    "ES617": "Málaga", "ES618": "Sevilla",
    "ES620": "Murcia",
    "ES630": "Ceuta",
    "ES640": "Melilla",
    "ES703": "El Hierro", "ES704": "Fuerteventura", "ES705": "Gran Canaria",
    "ES706": "La Gomera", "ES707": "La Palma", "ES708": "Lanzarote",
    "ES709": "Tenerife",
    "ES1": "Noroeste (España)", "ES2": "Noreste (España)",
    "ES3": "Comunidad de Madrid (España)", "ES4": "Centro (España)",
    "ES5": "Este (España)", "ES6": "Sur (España)",
    "ES7": "Canarias (España)"
}

TAMANO_LOTE = 10


# ============================================================
# FUNCIONES PARA FUENTES
# ============================================================

def normalizar_fuentes(fuente):
    """
    Convierte la cadena de fuentes en una lista limpia.
    Ejemplo:
        'TED, Licitaciones Generales PLACSP'
    ->
        ['TED', 'Licitaciones Generales PLACSP']
    """
    if not fuente:
        return []

    return [
        f.strip()
        for f in str(fuente).split(",")
        if f.strip()
    ]


def contiene_fuente(fuente, nombre_fuente):
    """
    Comprueba si una fuente concreta está presente,
    independientemente de mayúsculas/minúsculas.
    """
    fuentes = normalizar_fuentes(fuente)

    return any(
        f.casefold() == nombre_fuente.casefold()
        for f in fuentes
    )


def quitar_fuente(fuente, nombre_fuente):
    """
    Elimina únicamente la fuente indicada.
    """
    fuentes = normalizar_fuentes(fuente)

    fuentes_restantes = [
        f
        for f in fuentes
        if f.casefold() != nombre_fuente.casefold()
    ]

    return ", ".join(fuentes_restantes)


# ============================================================
# FUNCIONES TED
# ============================================================

def mapear_lugar(lugar_str):
    if not lugar_str or lugar_str == "No especificado":
        return lugar_str

    elementos = [
        e.strip()
        for e in lugar_str.split(",")
    ]

    elementos_mapeados = [
        MAPEO_NUTS_TED.get(el, el)
        for el in elementos
    ]

    return ", ".join(
        dict.fromkeys(elementos_mapeados)
    )


def procesar_campo(campo, es_lista=False):

    if isinstance(campo, list):

        limpios = [
            str(x)
            for x in campo
            if x
        ]

        if es_lista:
            return list(
                dict.fromkeys(limpios)
            )

        return (
            limpios[0]
            if limpios
            else "No especificado"
        )

    elif isinstance(campo, dict):

        return (
            campo.get("eng")
            or next(
                iter(campo.values()),
                "No especificado"
            )
        )

    return (
        str(campo)
        if campo
        else "No especificado"
    )


# ============================================================
# AUDITORÍA TED
# ============================================================

def auditar_licitaciones_abiertas_ted():

    hoy = date.today()

    url_api = (
        "https://api.ted.europa.eu/v3/notices/search"
    )

    fields_solicitados = [
        "publication-number",
        "contract-title",
        "notice-title",
        "organisation-name-buyer",
        "deadline-receipt-request",
        "place-of-performance",
        "classification-cpv",
        "description-proc",
        "total-value",
        "notice-type",
        "form-type"
    ]

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
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
        "licitaciones TED con fecha 'No especificada'...\n"
    )

    while True:

        # ====================================================
        # BUSCAR REGISTROS TED
        # ====================================================

        try:

            existentes_resp = (
                supabase
                .table("licitaciones")
                .select(
                    "id, enlace, titulo, fecha_fin, fuente"
                )
                .execute()
            )

            registros = []

            for reg in existentes_resp.data:

                if not contiene_fuente(
                    reg.get("fuente", ""),
                    "TED"
                ):
                    continue

                if reg.get("fecha_fin") != "No especificada":
                    continue

                if reg.get("id") in ids_procesados:
                    continue

                registros.append(reg)

                if len(registros) >= TAMANO_LOTE:
                    break

        except Exception as e:

            print(
                f"❌ Error al consultar Supabase: {e}"
            )
            break

        if not registros:

            print(
                "\n🎉 ¡Proceso finalizado! "
                "No quedan más licitaciones TED "
                "pendientes de auditar."
            )
            break

        print(
            f"\n📦 --- Procesando Lote TED "
            f"{lote_contador} "
            f"({len(registros)} registros) ---"
        )

        ids_a_borrar = []

        # ====================================================
        # PROCESAR LOTE
        # ====================================================

        for reg in registros:

            rec_id = reg.get("id")

            ids_procesados.append(rec_id)

            enlace = reg.get(
                "enlace",
                ""
            )

            titulo = reg.get(
                "titulo",
                "Sin título"
            )

            if not enlace:
                continue

            # ------------------------------------------------
            # NÚMERO DE PUBLICACIÓN
            # ------------------------------------------------

            parts = enlace.split("/")

            pub_number = (
                parts[-1]
                if parts
                else None
            )

            if not pub_number:
                continue

            payload = {
                "query": (
                    f"publication-number = '{pub_number}'"
                ),
                "fields": fields_solicitados,
                "limit": 1
            }

            # =================================================
            # CONSULTAR TED
            # =================================================

            try:

                resp = requests.post(
                    url_api,
                    json=payload,
                    headers=headers,
                    timeout=15
                )

                if resp.status_code != 200:

                    print(
                        f"   ⚠️ Error API TED "
                        f"HTTP {resp.status_code} "
                        f"para {pub_number}"
                    )

                    total_errores += 1
                    continue

                data = resp.json()

                notices = data.get(
                    "notices",
                    []
                )

                # =============================================
                # YA NO EXISTE EN TED
                # =============================================

                if not notices:

                    fuente_actual = str(
                        reg.get(
                            "fuente",
                            ""
                        )
                    )

                    nueva_fuente = quitar_fuente(
                        fuente_actual,
                        "TED"
                    )

                    if not nueva_fuente:

                        ids_a_borrar.append(
                            rec_id
                        )

                        print(
                            f"   🗑️ [A BORRAR - "
                            f"Ya no está en TED]: "
                            f"{titulo[:50]}..."
                        )

                        total_eliminadas += 1

                    else:

                        try:

                            (
                                supabase
                                .table("licitaciones")
                                .update({
                                    "fuente": nueva_fuente
                                })
                                .eq(
                                    "id",
                                    rec_id
                                )
                                .execute()
                            )

                            print(
                                f"   🔄 [TED ELIMINADO "
                                f"DE FUENTE]: "
                                f"{titulo[:50]}..."
                            )

                            total_fuentes_quitadas += 1

                        except Exception as e:

                            print(
                                f"   ⚠️ Error quitando "
                                f"TED de fuente "
                                f"{pub_number}: {e}"
                            )

                            total_errores += 1

                    continue

                aviso = notices[0]

                n_type = str(
                    aviso.get(
                        "notice-type",
                        ""
                    )
                ).lower()

                f_type = str(
                    aviso.get(
                        "form-type",
                        ""
                    )
                ).lower()

                # =================================================
                # COMPROBAR SI ESTÁ ADJUDICADA/CERRADA
                # =================================================

                es_adjudicado = (
                    n_type.startswith("can-")
                    or "award" in n_type
                    or f_type == "result"
                )

                es_veat = (
                    n_type.startswith(
                        "dir-awa-pre"
                    )
                    or "dir-awa-pre" in n_type
                    or "veat" in n_type
                )

                if es_adjudicado or es_veat:

                    fuente_actual = str(
                        reg.get(
                            "fuente",
                            ""
                        )
                    )

                    nueva_fuente = quitar_fuente(
                        fuente_actual,
                        "TED"
                    )

                    if not nueva_fuente:

                        ids_a_borrar.append(
                            rec_id
                        )

                        print(
                            f"   🗑️ [A BORRAR - "
                            f"Adjudicada/Cerrada en TED]: "
                            f"{titulo[:50]}..."
                        )

                        total_eliminadas += 1

                    else:

                        try:

                            (
                                supabase
                                .table("licitaciones")
                                .update({
                                    "fuente": nueva_fuente
                                })
                                .eq(
                                    "id",
                                    rec_id
                                )
                                .execute()
                            )

                            print(
                                f"   🔄 [TED ELIMINADO "
                                f"POR CIERRE]: "
                                f"{titulo[:50]}..."
                            )

                            total_fuentes_quitadas += 1

                        except Exception as e:

                            print(
                                f"   ⚠️ Error quitando "
                                f"TED por cierre "
                                f"{pub_number}: {e}"
                            )

                            total_errores += 1

                    continue

                # =================================================
                # COMPROBAR FECHA DE FIN
                # =================================================

                fechas_cierre = procesar_campo(
                    aviso.get(
                        "deadline-receipt-request"
                    ),
                    es_lista=True
                )

                nueva_fecha_fin = (
                    "No especificada"
                )

                if fechas_cierre != "No especificado":

                    limite_str = (
                        fechas_cierre[0]
                        if isinstance(
                            fechas_cierre,
                            list
                        )
                        else str(
                            fechas_cierre
                        )
                    )

                    nueva_fecha_fin = (
                        limite_str[:10]
                    )

                # =================================================
                # COMPROBAR SI YA HA CADUCADO
                # =================================================

                if (
                    nueva_fecha_fin
                    != "No especificada"
                ):

                    try:

                        f_fin_date = (
                            datetime.strptime(
                                nueva_fecha_fin,
                                "%Y-%m-%d"
                            ).date()
                        )

                        if f_fin_date < hoy:

                            fuente_actual = str(
                                reg.get(
                                    "fuente",
                                    ""
                                )
                            )

                            nueva_fuente = (
                                quitar_fuente(
                                    fuente_actual,
                                    "TED"
                                )
                            )

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

                            else:

                                try:

                                    (
                                        supabase
                                        .table("licitaciones")
                                        .update({
                                            "fuente": nueva_fuente
                                        })
                                        .eq(
                                            "id",
                                            rec_id
                                        )
                                        .execute()
                                    )

                                    print(
                                        f"   🔄 [TED ELIMINADO "
                                        f"POR CADUCIDAD]: "
                                        f"{titulo[:50]}..."
                                    )

                                    total_fuentes_quitadas += 1

                                except Exception as e:

                                    print(
                                        f"   ⚠️ Error quitando "
                                        f"TED por caducidad "
                                        f"{pub_number}: {e}"
                                    )

                                    total_errores += 1

                            continue

                    except ValueError:
                        pass

                # =================================================
                # ACTUALIZAR FECHA DE FIN
                # =================================================

                if (
                    nueva_fecha_fin
                    != reg.get("fecha_fin")
                ):

                    try:

                        (
                            supabase
                            .table("licitaciones")
                            .update({
                                "fecha_fin": nueva_fecha_fin,
                                "es_actualizada": True
                            })
                            .eq(
                                "id",
                                rec_id
                            )
                            .execute()
                        )

                        print(
                            f"   🔄 [ACTUALIZADA "
                            f"Fecha Fin TED a "
                            f"{nueva_fecha_fin}]: "
                            f"{titulo[:50]}..."
                        )

                        total_actualizadas += 1

                    except Exception as e:

                        print(
                            f"   ⚠️ Error actualizando "
                            f"fecha TED {pub_number}: {e}"
                        )

                        total_errores += 1

                else:

                    total_sin_cambios += 1

                time.sleep(0.2)

            except Exception as e:

                print(
                    f"   ⚠️ Error procesando TED "
                    f"{pub_number}: {e}"
                )

                total_errores += 1
                continue

        # ====================================================
        # BORRADO EN BLOQUE
        # ====================================================

        if ids_a_borrar:

            try:

                (
                    supabase
                    .table("licitaciones")
                    .delete()
                    .in_(
                        "id",
                        ids_a_borrar
                    )
                    .execute()
                )

                print(
                    f"   🗑️ -> ¡{len(ids_a_borrar)} "
                    f"licitaciones TED eliminadas "
                    f"de Supabase en este lote!"
                )

            except Exception as e:

                print(
                    f"   ❌ Error al eliminar "
                    f"lote TED en Supabase: {e}"
                )

                total_errores += 1

        lote_contador += 1

        time.sleep(0.5)

    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n" + "=" * 50)

    print(
        "📊 RESUMEN FINAL AUDITORÍA TED:"
    )

    print(
        f"  - Eliminadas "
        f"(TED única fuente): "
        f"{total_eliminadas}"
    )

    print(
        f"  - TED eliminado de "
        f"fuentes combinadas: "
        f"{total_fuentes_quitadas}"
    )

    print(
        f"  - Actualizadas "
        f"(nueva fecha): "
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
        "✅ ¡Auditoría TED completada!"
    )


if __name__ == "__main__":
    auditar_licitaciones_abiertas_ted()
