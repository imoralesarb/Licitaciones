from datetime import datetime, date
import os
import time
from sodapy import Socrata
from supabase import Client, create_client

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TAMANO_LOTE = 10

print("Iniciando auditoría de licitaciones PSCP Catalunya...")


def quitar_fuente_pscp(fuente):
    """
    Elimina únicamente 'PSCP Catalunya' de una cadena de fuentes.
    """
    if not fuente:
        return ""

    fuentes = [
        f.strip()
        for f in str(fuente).split(",")
        if f.strip()
    ]

    fuentes_restantes = [
        f
        for f in fuentes
        if f.casefold() != "pscp catalunya"
    ]

    return ", ".join(fuentes_restantes)


def auditar_licitaciones_abiertas_pscp():
    hoy = date.today()
    dataset_id = "ybgg-dgi6"

    client = Socrata(
        "analisi.transparenciacatalunya.cat",
        None
    )

    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    lote_contador = 1

    print(
        "\nIniciando auditoría por lotes de licitaciones "
        "PSCP Catalunya con fecha_fin 'No especificada'...\n"
    )

    while True:

        # ========================================================
        # OBTENER LOTE DE LICITACIONES PENDIENTES
        # ========================================================

        try:
            response = (
                supabase
                .table("licitaciones")
                .select(
                    "id, enlace, titulo, fuente, fecha_fin, "
                    "es_actualizada, texto_completo"
                )
                .ilike("fuente", "%PSCP Catalunya%")
                .eq("fecha_fin", "No especificada")
                .limit(TAMANO_LOTE)
                .execute()
            )

            registros = response.data

        except Exception as e:
            print(f"Error al consultar Supabase: {e}")
            break

        if not registros:
            print(
                "\n¡Proceso finalizado! "
                "No quedan más licitaciones PSCP Catalunya "
                "pendientes de auditar."
            )
            break

        print(
            f"\n--- Procesando Lote PSCP {lote_contador} "
            f"({len(registros)} registros) ---"
        )

        ids_a_borrar = []

        # ========================================================
        # PROCESAR REGISTROS
        # ========================================================

        for reg in registros:
            rec_id = reg.get("id")
            enlace = reg.get("enlace", "")
            titulo = reg.get("titulo", "Sin título")

            if not enlace:
                continue

            # ----------------------------------------------------
            # CONSULTAR LA API DE PSCP
            # ----------------------------------------------------

            query_socrata = (
                f"enllac_publicacio like '%{enlace}%'"
            )

            try:
                chunk = client.get(
                    dataset_id,
                    where=query_socrata,
                    limit=1
                )

                # ------------------------------------------------
                # YA NO EXISTE EN PSCP
                # ------------------------------------------------

                if not chunk:
                    fuente_actual = reg.get("fuente", "")
                    nueva_fuente = quitar_fuente_pscp(
                        fuente_actual
                    )

                    # Si no queda ninguna fuente, eliminar registro
                    if not nueva_fuente:
                        ids_a_borrar.append(rec_id)

                        print(
                            f"   [A BORRAR - Ya no está en PSCP]: "
                            f"{titulo[:50]}..."
                        )

                    # Si tiene otras fuentes, conservar registro
                    else:
                        (
                            supabase
                            .table("licitaciones")
                            .update({
                                "fuente": nueva_fuente
                            })
                            .eq("id", rec_id)
                            .execute()
                        )

                        print(
                            f"   [PSCP ELIMINADO DE FUENTE - "
                            f"Se conservan otras fuentes]: "
                            f"{titulo[:50]}..."
                        )

                    total_eliminadas += 1
                    continue

                aviso = chunk[0]

                # ------------------------------------------------
                # COMPROBAR FASE
                # ------------------------------------------------

                fase = str(
                    aviso.get("fase_publicacio", "")
                ).strip().lower()

                if (
                    fase
                    and fase != "anunci de licitació"
                ):
                    fuente_actual = reg.get("fuente", "")
                    nueva_fuente = quitar_fuente_pscp(
                        fuente_actual
                    )

                    if not nueva_fuente:
                        ids_a_borrar.append(rec_id)

                        print(
                            f"   [A BORRAR - Cambio de fase "
                            f"a '{fase}']: {titulo[:50]}..."
                        )

                    else:
                        (
                            supabase
                            .table("licitaciones")
                            .update({
                                "fuente": nueva_fuente
                            })
                            .eq("id", rec_id)
                            .execute()
                        )

                        print(
                            f"   [PSCP ELIMINADO - Cambio de fase "
                            f"a '{fase}', se conservan otras fuentes]: "
                            f"{titulo[:50]}..."
                        )

                    total_eliminadas += 1
                    continue

                # ------------------------------------------------
                # COMPROBAR FECHA DE FIN
                # ------------------------------------------------

                fecha_cierre_raw = aviso.get(
                    "termini_presentacio_ofertes"
                )

                nueva_fecha_fin = "No especificada"

                if fecha_cierre_raw:
                    nueva_fecha_fin = str(
                        fecha_cierre_raw
                    )[:10]

                # ------------------------------------------------
                # SI LA LICITACIÓN YA HA CADUCADO
                # ------------------------------------------------

                if nueva_fecha_fin != "No especificada":
                    try:
                        f_fin_date = datetime.strptime(
                            nueva_fecha_fin,
                            "%Y-%m-%d"
                        ).date()

                        if f_fin_date < hoy:
                            fuente_actual = reg.get(
                                "fuente", ""
                            )

                            nueva_fuente = quitar_fuente_pscp(
                                fuente_actual
                            )

                            if not nueva_fuente:
                                ids_a_borrar.append(rec_id)

                                print(
                                    f"   [A BORRAR - Caducada]: "
                                    f"{titulo[:50]}..."
                                )

                            else:
                                (
                                    supabase
                                    .table("licitaciones")
                                    .update({
                                        "fuente": nueva_fuente
                                    })
                                    .eq("id", rec_id)
                                    .execute()
                                )

                                print(
                                    f"   [PSCP ELIMINADO - Caducada, "
                                    f"se conservan otras fuentes]: "
                                    f"{titulo[:50]}..."
                                )

                            total_eliminadas += 1
                            continue

                    except ValueError:
                        pass

                # ------------------------------------------------
                # ACTUALIZAR FECHA DE FIN
                # ------------------------------------------------

                fecha_fin_actual = reg.get("fecha_fin")

                if nueva_fecha_fin != fecha_fin_actual:
                    actualizar_datos = {
                        "fecha_fin": nueva_fecha_fin,
                        "es_actualizada": True
                    }

                    (
                        supabase
                        .table("licitaciones")
                        .update(actualizar_datos)
                        .eq("id", rec_id)
                        .execute()
                    )

                    print(
                        f"    [ACTUALIZADA Fecha Fin PSCP a "
                        f"{nueva_fecha_fin}]: "
                        f"{titulo[:50]}..."
                    )

                    total_actualizadas += 1

                else:
                    total_sin_cambios += 1

                time.sleep(0.2)

            except Exception as e:
                print(
                    f"    Error procesando registro PSCP "
                    f"{enlace}: {e}"
                )
                continue

        # ========================================================
        # BORRADO EN BLOQUE
        # ========================================================

        if ids_a_borrar:
            try:
                (
                    supabase
                    .table("licitaciones")
                    .delete()
                    .in_("id", ids_a_borrar)
                    .execute()
                )

                print(
                    f"    -> {len(ids_a_borrar)} licitaciones "
                    f"eliminadas de Supabase en este lote."
                )

            except Exception as e:
                print(
                    f"    Error al eliminar lote en Supabase: {e}"
                )

        lote_contador += 1
        time.sleep(0.5)

    # ============================================================
    # RESUMEN
    # ============================================================

    print("\n" + "=" * 50)
    print(" RESUMEN FINAL AUDITORÍA PSCP CATALUNYA:")
    print(
        f"  - Eliminadas / PSCP retirada: "
        f"{total_eliminadas}"
    )
    print(
        f"  - Actualizadas (nueva fecha): "
        f"{total_actualizadas}"
    )
    print(
        f"  - Sin cambios: "
        f"{total_sin_cambios}"
    )
    print(
        " ¡Auditoría de licitaciones PSCP Catalunya "
        "completada con éxito!"
    )


if __name__ == "__main__":
    auditar_licitaciones_abiertas_pscp()
