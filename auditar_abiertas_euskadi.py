# Editado 10/09/2026
# Audita las licitaciones de Euskadi almacenadas en Supabase mediante scraping HTML,
# comprobando la fecha de cierre y el estado de cada licitación. Elimina las que
# ya no permiten presentar ofertas y actualiza la fecha de fin cuando se obtiene
# nueva información. No modifica las banderas es_novedad ni es_actualizada.

from datetime import datetime, date
import os
import time
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client

# ========================================================
# CONFIGURACIÓN SUPABASE Y GENERAL
# ========================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

TAMANO_LOTE = 50


# ========================================================
# EXTRAER FECHA DE UN TEXTO
# ========================================================

def extraer_fecha(texto):
    if not texto:
        return None
    texto = str(texto).strip()
    formatos = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y"
    ]
    for formato in formatos:
        try:
            return datetime.strptime(
                texto,
                formato
            ).date()
        except ValueError:
            continue
    return None


# ========================================================
# EXTRAER FECHA DE FIN DEL HTML
# ========================================================

def extraer_fecha_fin_html(soup):
    # Buscar campos <dt> / <dd>

    dts = soup.find_all("dt")
    for dt in dts:
        texto_dt = dt.get_text(
            " ",
            strip=True
        ).lower()

        if (
            "fecha fin" in texto_dt
            or "fecha de fin" in texto_dt
            or "fecha límite" in texto_dt
            or "fecha limite" in texto_dt
            or "plazo de presentación" in texto_dt
            or "plazo de presentacion" in texto_dt
        ):
            dd = dt.find_next_sibling("dd")
            if dd:
                texto_dd = dd.get_text(
                    " ",
                    strip=True
                )
                fecha = extraer_fecha(
                    texto_dd
                )
                if fecha:
                    return fecha


    # ----------------------------------------------------
    # Buscar directamente textos dentro de <dd>
    # por si la estructura HTML es ligeramente diferente
    # ----------------------------------------------------

    for dd in soup.find_all("dd"):
        texto_dd = dd.get_text(
            " ",
            strip=True
        )
        fecha = extraer_fecha(
            texto_dd
        )
        if fecha:
            return fecha
    return None


# ========================================================
# DETECTAR ESTADO DE TRAMITACIÓN
# ========================================================

def detectar_estado_html(soup):

    """
    Extrae el estado de tramitación de la licitación
    buscando específicamente el <dd> asociado al <dt>
    "Estado de tramitación".

    Devuelve:

        estado, debe_eliminar

    Estados que se conservan:

        - Alerta temprana
        - Abierto / Plazo de presentación
        - Suspensión por recurso

    Estados que se eliminan:

        - Plazo cerrado
        - Adjudicación
        - Desistimiento / Renuncia
        - Histórico
        - Formalizado / En ejecución
        - Desierto
        - Modificación contrato
        - En redacción
        - Anulado
        - Finalizado
    """

    estado = "No determinado"


    # ----------------------------------------------------
    # Buscar el <dt> correspondiente al estado
    # ----------------------------------------------------

    dts = soup.find_all("dt")
    for dt in dts:
        texto_dt = dt.get_text(
            " ",
            strip=True
        ).lower()

        if (
            "estado de tramitación" in texto_dt
            or "estado tramitación" in texto_dt
            or "estado de tramitacion" in texto_dt
            or "estado tramitacion" in texto_dt
        ):
            dd = dt.find_next_sibling("dd")
            if dd:
                estado = dd.get_text(
                    " ",
                    strip=True
                )
                estado_lower = estado.lower()


                # ----------------------------------------
                # ESTADOS QUE IMPLICAN ELIMINACIÓN
                # ----------------------------------------

                estados_eliminar = [
                    "plazo cerrado",
                    "adjudicación",
                    "adjudicacion",
                    "desistimiento / renuncia",
                    "desistimiento",
                    "renuncia",
                    "histórico",
                    "historico",
                    "formalizado / en ejecución",
                    "formalizado / en ejecucion",
                    "formalizado",
                    "en ejecución",
                    "en ejecucion",
                    "desierto",
                    "modificación contrato",
                    "modificacion contrato",
                    "en redacción",
                    "en redaccion",
                    "anulado",
                    "finalizado"
                ]

                for estado_eliminar in estados_eliminar:
                    if estado_eliminar in estado_lower:
                        return (
                            estado,
                            True
                        )

                # ----------------------------------------
                # ESTADOS QUE SE CONSERVAN
                # ----------------------------------------

                estados_mantener = [
                    "alerta temprana",
                    "abierto / plazo de presentación",
                    "abierto / plazo de presentacion",
                    "suspensión por recurso",
                    "suspension por recurso"
                ]
                for estado_mantener in estados_mantener:
                    if estado_mantener in estado_lower:
                        return (
                            estado,
                            False
                        )

                # ----------------------------------------
                # Si aparece un estado desconocido,
                # NO eliminamos automáticamente.
                # ----------------------------------------

                return (
                    estado,
                    False
                )
    return (
        estado,
        False
    )

# ========================================================
# AUDITORÍA
# ========================================================

def auditar_licitaciones_html_euskadi():
    hoy = date.today()
    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    total_errores = 0

    print(
        "Iniciando auditoría mediante scraping HTML de Euskadi...\n"
    )

    # ====================================================
    # 1. OBTENER TODAS LAS LICITACIONES DE EUSKADI
    # ====================================================

    try:
        response = (
            supabase
            .table("licitaciones")
            .select(
                "id, titulo, enlace, fecha_fin, "
                "es_novedad, es_actualizada"
            )
            .eq("fuente", "Euskadi")
            .execute()
        )

        registros = response.data or []
    except Exception as e:
        print(
            f"Error al consultar Supabase: {e}"
        )

        return

    print(
        f"Se han encontrado "
        f"{len(registros)} licitaciones de Euskadi para auditar."
    )

    if not registros:
        print(
            "\nNo hay licitaciones de Euskadi para auditar."
        )
        return

    # ====================================================
    # 2. PROCESAR EN LOTES DE 50
    # ====================================================

    total_registros = len(registros)
    for inicio in range(
        0,
        total_registros,
        TAMANO_LOTE
    ):
        lote = registros[
            inicio:inicio + TAMANO_LOTE
        ]
        numero_lote = (
            inicio // TAMANO_LOTE
        ) + 1

        total_lotes = (
            (total_registros + TAMANO_LOTE - 1)
            // TAMANO_LOTE
        )

        print(
            f"\n--- Procesando Lote "
            f"{numero_lote}/{total_lotes} "
            f"({len(lote)} registros) ---"
        )

        ids_a_borrar = []

        # =================================================
        # 3. PROCESAR CADA LICITACIÓN
        # =================================================

        for reg in lote:
            rec_id = reg.get("id")
            enlace = str(
                reg.get("enlace") or ""
            ).strip()
            titulo = str(
                reg.get("titulo") or "Sin título"
            ).strip()

            fecha_fin_actual = str(
                reg.get("fecha_fin") or ""
            ).strip()

            # ---------------------------------------------
            # Sin enlace
            # ---------------------------------------------

            if not enlace:
                ids_a_borrar.append(
                    rec_id
                )
                total_eliminadas += 1
                print(
                    f"    [A BORRAR - Sin enlace] "
                    f"{titulo[:60]}..."
                )
                continue

            # ---------------------------------------------
            # Comprobar fecha_fin que ya tenemos
            # ---------------------------------------------

            fecha_fin_db = None
            if (
                fecha_fin_actual
                and fecha_fin_actual != "No especificada"
            ):
                fecha_fin_db = extraer_fecha(
                    fecha_fin_actual
                )


            # =================================================
            # DESCARGAR HTML
            # =================================================

            try:
                headers = {
                    "User-Agent":
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/140.0 Safari/537.36"
                }
                resp = requests.get(
                    enlace,
                    headers=headers,
                    timeout=10
                )

                # ---------------------------------------------
                # Error HTTP
                # ---------------------------------------------

                if resp.status_code != 200:
                    total_errores += 1
                    print(
                        f"    [ERROR HTTP "
                        f"{resp.status_code}] "
                        f"{titulo[:60]}..."
                    )
                    continue
                soup = BeautifulSoup(
                    resp.text,
                    "html.parser"
                )


                # =================================================
                # 4. EXTRAER FECHA DEL HTML
                # =================================================

                fecha_fin_html = (
                    extraer_fecha_fin_html(
                        soup
                    )
                )


                # =================================================
                # 5. ACTUALIZAR FECHA SI AHORA LA CONOCEMOS
                # =================================================

                if (
                    fecha_fin_db is None
                    and fecha_fin_html is not None
                ):

                    fecha_fin_nueva = (
                        fecha_fin_html.strftime(
                            "%Y-%m-%d"
                        )
                    )


                    try:
                        supabase \
                            .table("licitaciones") \
                            .update({
                                "fecha_fin":
                                    fecha_fin_nueva
                            }) \
                            .eq(
                                "id",
                                rec_id
                            ) \
                            .execute()

                        total_actualizadas += 1

                        print(
                            f"    [FECHA ACTUALIZADA] "
                            f"{titulo[:50]}... "
                            f"-> {fecha_fin_nueva}"
                        )

                        # Usamos esta fecha también
                        # para comprobar si ha caducado.

                        fecha_fin_db = (
                            fecha_fin_html
                        )


                    except Exception as e:
                        print(
                            f"    Error actualizando "
                            f"fecha_fin para ID "
                            f"{rec_id}: {e}"
                        )


                # =================================================
                # 6. COMPROBAR SI LA FECHA YA HA PASADO
                # =================================================

                if fecha_fin_db is not None:
                    if fecha_fin_db < hoy:
                        ids_a_borrar.append(
                            rec_id
                        )
                        total_eliminadas += 1

                        print(
                            f"    [A BORRAR - "
                            f"FECHA CADUCADA] "
                            f"{titulo[:50]}... "
                            f"-> fecha fin: "
                            f"{fecha_fin_db}"
                        )

                        continue


                # =================================================
                # 7. COMPROBAR ESTADO DE TRAMITACIÓN
                # =================================================

                estado, debe_eliminar = (
                    detectar_estado_html(
                        soup
                    )
                )


                # ---------------------------------------------
                # Estado que implica eliminación
                # ---------------------------------------------

                if debe_eliminar:
                    ids_a_borrar.append(
                        rec_id
                    )
                    total_eliminadas += 1
                    print(
                        f"    [A BORRAR - ESTADO] "
                        f"{titulo[:50]}..."
                    )
                    print(
                        f"      -> Estado: "
                        f"{estado}"
                    )
                    continue


                # ---------------------------------------------
                # Registro válido
                # ---------------------------------------------

                total_sin_cambios += 1
                print(
                    f"    [OK] "
                    f"{titulo[:50]}..."
                )
                print(
                    f"      -> Estado: "
                    f"{estado}"
                )
                if fecha_fin_db:
                    print(
                        f"      -> Fecha fin: "
                        f"{fecha_fin_db}"
                    )

                else:
                    print(
                        "      -> Fecha fin: "
                        "No especificada"
                    )

                time.sleep(0.2)
            except requests.exceptions.Timeout:

                total_errores += 1

                print(
                    f"    [ERROR TIMEOUT] "
                    f"{titulo[:60]}..."
                )

                continue
            except Exception as e:
                total_errores += 1
                print(
                    f"    Error procesando "
                    f"enlace {enlace}: {e}"
                )
                continue


        # =================================================
        # 8. ELIMINAR LICITACIONES DEL LOTE
        # =================================================

        if ids_a_borrar:
            try:
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
                    f"    -> ¡{len(ids_a_borrar)} "
                    f"licitaciones eliminadas "
                    f"de Supabase!"
                )
            except Exception as e:
                print(
                    f"    Error al eliminar "
                    f"en Supabase: {e}"
                )
        time.sleep(0.2)


    # ====================================================
    # 9. RESUMEN FINAL
    # ====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "AUDITORÍA FINALIZADA"
    )

    print(
        "=" * 60
    )

    print(
        f"Total auditadas: "
        f"{total_registros}"
    )

    print(
        f"Eliminadas: "
        f"{total_eliminadas}"
    )

    print(
        f"Fechas actualizadas: "
        f"{total_actualizadas}"
    )

    print(
        f"Sin cambios: "
        f"{total_sin_cambios}"
    )

    print(
        f"Errores: "
        f"{total_errores}"
    )

    print(
        "=" * 60
    )


# ========================================================
# EJECUCIÓN
# ========================================================

if __name__ == "__main__":

    auditar_licitaciones_html_euskadi()
```
