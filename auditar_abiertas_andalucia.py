# -*- coding: utf-8 -*-

# Editado 11/09/2026
#
# Audita las licitaciones de Andalucía almacenadas en Supabase mediante
# scraping HTML, comprobando la fecha límite y el estado del procedimiento.
#
# Si una licitación ya no está abierta:
#    - Si Andalucía es la única fuente -> elimina la licitación.
#    - Si tiene otras fuentes -> elimina únicamente "Andalucía" de la columna fuente.
#
# Si se obtiene una nueva fecha límite, se actualiza fecha_fin.
#
# No modifica las banderas es_novedad ni es_actualizada.

from datetime import datetime, date
import os
import time
import re
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

FUENTE_ANDALUCIA = "Andalucía"

TAMANO_LOTE = 50

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
}


# ========================================================
# EXTRAER FECHA DE UN TEXTO
# ========================================================

def extraer_fecha(texto):

    if not texto:
        return None

    texto = str(texto).strip()

    # ----------------------------------------------------
    # Primero buscamos fechas dentro de textos más largos.
    # ----------------------------------------------------

    patrones = [
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b",
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"
    ]

    for patron in patrones:

        coincidencia = re.search(
            patron,
            texto
        )

        if coincidencia:

            try:

                grupos = coincidencia.groups()

                # Formato YYYY-MM-DD
                if len(grupos[0]) == 4:

                    return date(
                        int(grupos[0]),
                        int(grupos[1]),
                        int(grupos[2])
                    )

                # Formato DD/MM/YYYY o DD-MM-YYYY
                return date(
                    int(grupos[2]),
                    int(grupos[1]),
                    int(grupos[0])
                )

            except ValueError:
                continue

    # ----------------------------------------------------
    # Por si el texto contiene únicamente una fecha
    # ----------------------------------------------------

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
# EXTRAER FECHA LÍMITE DEL HTML DE ANDALUCÍA
# ========================================================

def extraer_fecha_fin_html(soup):

    # ====================================================
    # 1. BUSCAR EN ESTRUCTURAS COMUNES (DT/DD, TH/TD, SPAN, ETC.)
    # ====================================================

    for etiqueta in soup.find_all(["dt", "th", "span", "div", "p", "td"]):

        texto_etiqueta = etiqueta.get_text(
            " ",
            strip=True
        ).lower()

        if (
            "fecha límite" in texto_etiqueta
            or "fecha limite" in texto_etiqueta
            or "data límite" in texto_etiqueta
            or "fin de plazo" in texto_etiqueta
            or "fecha de presentación" in texto_etiqueta
        ):

            # Intentar buscar en el siguiente hermano o dentro de la misma etiqueta
            siguiente = etiqueta.find_next_sibling(["dd", "td", "span", "div"])
            if siguiente:
                texto_siguiente = siguiente.get_text(" ", strip=True)
                fecha = extraer_fecha(texto_siguiente)
                if fecha:
                    return fecha

            # Si no hay hermano, buscar fecha en el propio texto de la etiqueta
            fecha = extraer_fecha(texto_etiqueta)
            if fecha:
                return fecha

    # ====================================================
    # 2. BÚSQUEDA DE SEGURIDAD EN TODO EL TEXTO HTML
    # ====================================================

    texto_completo = soup.get_text(
        " ",
        strip=True
    )

    patrones = [
        r"fecha\s+l[ií]mite.{0,100}",
        r"fin\s+de\s+plazo.{0,100}"
    ]

    for patron in patrones:

        coincidencia = re.search(
            patron,
            texto_completo,
            flags=re.IGNORECASE
        )

        if coincidencia:

            texto = coincidencia.group(0)
            fecha = extraer_fecha(texto)

            if fecha:
                return fecha

    return None


# ====================================================
# DETECTAR ESTADO DEL PROCEDIMIENTO DE ANDALUCÍA
# ====================================================

def detectar_estado_html(soup):

    """
    Extrae el estado del procedimiento de Andalucía.
    Devuelve:
        estado, debe_eliminar
    """

    estado = "No determinado"

    texto_pagina = soup.get_text(
        " ",
        strip=True
    )

    # ====================================================
    # 1. BUSCAR PATRÓN DE ESTADO EN EL TEXTO
    # ====================================================

    coincidencia = re.search(
        r"Estado\s+(?:del\s+)?procedimiento\s*:\s*([^|]+?)(?=\s{2,}|$)",
        texto_pagina,
        flags=re.IGNORECASE
    )

    if coincidencia:

        estado_extraido = coincidencia.group(1).strip()

        if estado_extraido:
            estado = estado_extraido

    # ====================================================
    # 2. BUSCAR EN ESTRUCTURAS ESPECÍFICAS (SPAN, P, DIV)
    # ====================================================

    for el in soup.find_all(["p", "div", "span", "td"]):

        texto_el = el.get_text(" ", strip=True)

        if "estado del procedimiento" in texto_el.lower() or "estado:" in texto_el.lower():
            partes = re.split(r"(?:estado\s+del\s+procedimiento|estado)\s*:", texto_el, flags=re.IGNORECASE)
            if len(partes) > 1:
                estado_extraido = partes[1].strip()
                if estado_extraido:
                    estado = estado_extraido
                break

    # ====================================================
    # NORMALIZAR ESTADO
    # ====================================================

    estado_lower = estado.lower().strip()

    # ====================================================
    # ESTADOS QUE IMPLICAN QUE YA NO ESTÁ ABIERTA
    # ====================================================

    estados_eliminar = [
        "cerrado",
        "cerrada",
        "adjudicado",
        "adjudicación",
        "adjudicacion",
        "desierto",
        "desierta",
        "desistimiento",
        "renuncia",
        "anulado",
        "anulada",
        "formalizado",
        "formalizada",
        "finalizado",
        "finalizada",
        "resuelto",
        "resuelta",
        "cancelado",
        "cancelada",
        "archivado",
        "archivada"
    ]

    for estado_eliminar in estados_eliminar:

        if estado_eliminar in estado_lower:

            return (
                estado,
                True
            )

    # ====================================================
    # ESTADOS QUE SE MANTIENEN
    # ====================================================

    estados_mantener = [
        "en curso",
        "abierto",
        "abierta",
        "publicada",
        "en plazo"
    ]

    for estado_mantener in estados_mantener:

        if estado_mantener in estado_lower:

            return (
                estado,
                False
            )

    # ====================================================
    # ESTADO DESCONOCIDO
    # Por seguridad NO eliminamos.
    # ====================================================

    return (
        estado,
        False
    )


# ========================================================
# NORMALIZAR FUENTES
# ========================================================

def obtener_fuentes(fuente_raw):

    if not fuente_raw:
        return []

    return [
        f.strip()
        for f in str(fuente_raw).split(",")
        if f.strip()
    ]


def construir_fuente(fuentes):

    fuentes_limpias = []

    for fuente in fuentes:

        fuente = str(
            fuente
        ).strip()

        if (
            fuente
            and fuente not in fuentes_limpias
        ):
            fuentes_limpias.append(
                fuente
            )

    return ", ".join(
        fuentes_limpias
    )


# ========================================================
# QUITAR ANDALUCÍA DE LA COLUMNA FUENTE
# ========================================================

def quitar_fuente_andalucia(fuente_raw):

    fuentes = obtener_fuentes(
        fuente_raw
    )

    fuentes = [
        fuente
        for fuente in fuentes
        if fuente.lower()
        != FUENTE_ANDALUCIA.lower()
    ]

    return construir_fuente(
        fuentes
    )


# ========================================================
# AUDITORÍA
# ========================================================

def auditar_licitaciones_html_andalucia():

    hoy = date.today()

    total_registros = 0
    total_eliminadas = 0
    total_fuente_quitada = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    total_errores = 0

    print(
        "Iniciando auditoría mediante "
        "scraping HTML de Andalucía...\n"
    )

    # ====================================================
    # 1. OBTENER LICITACIONES DE ANDALUCÍA
    # ====================================================

    try:

        response = (
            supabase
            .table("licitaciones")
            .select(
                "id, titulo, enlace, fecha_fin, fuente"
            )
            .ilike(
                "fuente",
                "%Andalucía%"
            )
            .execute()
        )

        registros = response.data or []

    except Exception as e:

        print(
            f"Error al consultar Supabase: {e}"
        )

        return

    # ====================================================
    # FILTRAR SOLO REGISTROS DONDE ANDALUCÍA
    # SEA REALMENTE UNA DE LAS FUENTES
    # ====================================================

    registros_andalucia = []

    for reg in registros:

        fuentes = obtener_fuentes(
            reg.get("fuente")
        )

        if any(
            fuente.lower()
            == FUENTE_ANDALUCIA.lower()
            for fuente in fuentes
        ):
            registros_andalucia.append(
                reg
            )

    registros = registros_andalucia

    total_registros = len(
        registros
    )

    print(
        f"Se han encontrado "
        f"{total_registros} licitaciones de Andalucía "
        f"para auditar."
    )

    if not registros:

        print(
            "\nNo hay licitaciones de Andalucía "
            "para auditar."
        )

        return

    # ====================================================
    # 2. PROCESAR EN LOTES
    # ====================================================

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
            (
                total_registros
                + TAMANO_LOTE
                - 1
            )
            // TAMANO_LOTE
        )

        print(
            f"\n--- Procesando Lote "
            f"{numero_lote}/{total_lotes} "
            f"({len(lote)} registros) ---"
        )

        # =================================================
        # 3. PROCESAR CADA LICITACIÓN
        # =================================================

        for reg in lote:

            rec_id = reg.get(
                "id"
            )

            enlace = str(
                reg.get("enlace") or ""
            ).strip()

            titulo = str(
                reg.get("titulo")
                or "Sin título"
            ).strip()

            fecha_fin_actual = str(
                reg.get("fecha_fin") or ""
            ).strip()

            fuente_actual = str(
                reg.get("fuente") or ""
            ).strip()

            # ---------------------------------------------
            # SIN ENLACE
            # ---------------------------------------------

            if not enlace:

                total_errores += 1

                print(
                    f"    [ERROR - Sin enlace] "
                    f"{titulo[:60]}..."
                )

                continue

            # =================================================
            # 4. COMPROBAR FECHA QUE YA TENEMOS
            # =================================================

            fecha_fin_db = None

            if (
                fecha_fin_actual
                and fecha_fin_actual
                != "No especificada"
            ):

                fecha_fin_db = extraer_fecha(
                    fecha_fin_actual
                )

            # =================================================
            # 5. DESCARGAR HTML
            # =================================================

            try:

                resp = requests.get(
                    enlace,
                    headers=HEADERS,
                    timeout=15
                )

                # ---------------------------------------------
                # ERROR HTTP
                # ---------------------------------------------

                if resp.status_code != 200:

                    total_errores += 1

                    print(
                        f"    [ERROR HTTP "
                        f"{resp.status_code}] "
                        f"{titulo[:60]}..."
                    )

                    continue

                # ---------------------------------------------
                # PARSEAR HTML
                # ---------------------------------------------

                soup = BeautifulSoup(
                    resp.content,
                    "html.parser"
                )

                # =================================================
                # 6. EXTRAER FECHA DEL HTML
                # =================================================

                fecha_fin_html = (
                    extraer_fecha_fin_html(
                        soup
                    )
                )

                # =================================================
                # 7. ACTUALIZAR FECHA SI ES NUEVA
                # =================================================

                if (
                    fecha_fin_html is not None
                    and (
                        fecha_fin_db is None
                        or fecha_fin_html
                        != fecha_fin_db
                    )
                ):

                    fecha_fin_nueva = (
                        fecha_fin_html.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    try:

                        (
                            supabase
                            .table("licitaciones")
                            .update({
                                "fecha_fin":
                                    fecha_fin_nueva
                            })
                            .eq(
                                "id",
                                rec_id
                            )
                            .execute()
                        )

                        total_actualizadas += 1

                        fecha_fin_db = (
                            fecha_fin_html
                        )

                        print(
                            f"    [FECHA ACTUALIZADA] "
                            f"{titulo[:50]}... "
                            f"-> {fecha_fin_nueva}"
                        )

                    except Exception as e:

                        total_errores += 1

                        print(
                            f"    [ERROR ACTUALIZANDO "
                            f"FECHA] ID {rec_id}: {e}"
                        )

                # =================================================
                # 8. COMPROBAR FECHA CADUCADA
                # =================================================

                if fecha_fin_db is not None:

                    if fecha_fin_db < hoy:

                        fuentes = obtener_fuentes(
                            fuente_actual
                        )

                        tiene_otra_fuente = any(
                            fuente.lower()
                            != FUENTE_ANDALUCIA.lower()
                            for fuente in fuentes
                        )

                        # -----------------------------------------
                        # ANDALUCÍA ES LA ÚNICA FUENTE
                        # -----------------------------------------

                        if not tiene_otra_fuente:

                            try:

                                (
                                    supabase
                                    .table("licitaciones")
                                    .delete()
                                    .eq(
                                        "id",
                                        rec_id
                                    )
                                    .execute()
                                )

                                total_eliminadas += 1

                                print(
                                    f"    [ELIMINADA - "
                                    f"FECHA CADUCADA] "
                                    f"{titulo[:50]}..."
                                )

                                print(
                                    f"      -> Fecha fin: "
                                    f"{fecha_fin_db}"
                                )

                            except Exception as e:

                                total_errores += 1

                                print(
                                    f"    [ERROR ELIMINANDO] "
                                    f"ID {rec_id}: {e}"
                                )

                        # -----------------------------------------
                        # HAY OTRA FUENTE
                        # -----------------------------------------

                        else:

                            nueva_fuente = (
                                quitar_fuente_andalucia(
                                    fuente_actual
                                )
                            )

                            try:

                                (
                                    supabase
                                    .table("licitaciones")
                                    .update({
                                        "fuente":
                                            nueva_fuente
                                    })
                                    .eq(
                                        "id",
                                        rec_id
                                    )
                                    .execute()
                                )

                                total_fuente_quitada += 1

                                print(
                                    f"    [ANDALUCÍA QUITADA - "
                                    f"FECHA CADUCADA] "
                                    f"{titulo[:50]}..."
                                )

                                print(
                                    f"      -> Fecha fin: "
                                    f"{fecha_fin_db}"
                                )

                                print(
                                    f"      -> Otras fuentes: "
                                    f"{nueva_fuente}"
                                )

                            except Exception as e:

                                total_errores += 1

                                print(
                                    f"    [ERROR QUITANDO "
                                    f"ANDALUCÍA] "
                                    f"ID {rec_id}: {e}"
                                )

                        continue

                # =================================================
                # 9. COMPROBAR ESTADO DEL PROCEDIMIENTO
                # =================================================

                estado, debe_eliminar = (
                    detectar_estado_html(
                        soup
                    )
                )

                # =================================================
                # 10. ESTADO QUE IMPLICA QUE ANDALUCÍA
                # YA NO ESTÁ ABIERTA
                # =================================================

                if debe_eliminar:

                    fuentes = obtener_fuentes(
                        fuente_actual
                    )

                    tiene_otra_fuente = any(
                        fuente.lower()
                        != FUENTE_ANDALUCIA.lower()
                        for fuente in fuentes
                    )

                    # -----------------------------------------
                    # ANDALUCÍA ES LA ÚNICA FUENTE
                    # -----------------------------------------

                    if not tiene_otra_fuente:

                        try:

                            (
                                supabase
                                .table("licitaciones")
                                .delete()
                                .eq(
                                    "id",
                                    rec_id
                                )
                                .execute()
                            )

                            total_eliminadas += 1

                            print(
                                f"    [ELIMINADA - "
                                f"ESTADO] "
                                f"{titulo[:50]}..."
                            )

                            print(
                                f"      -> Estado: "
                                f"{estado}"
                            )

                        except Exception as e:

                            total_errores += 1

                            print(
                                f"    [ERROR ELIMINANDO] "
                                f"ID {rec_id}: {e}"
                            )

                    # -----------------------------------------
                    # HAY OTRA FUENTE
                    # -----------------------------------------

                    else:

                        nueva_fuente = (
                            quitar_fuente_andalucia(
                                fuente_actual
                            )
                        )

                        try:

                            (
                                supabase
                                .table("licitaciones")
                                .update({
                                    "fuente":
                                        nueva_fuente
                                })
                                .eq(
                                    "id",
                                    rec_id
                                )
                                .execute()
                            )

                            total_fuente_quitada += 1

                            print(
                                f"    [ANDALUCÍA QUITADA - "
                                f"ESTADO] "
                                f"{titulo[:50]}..."
                            )

                            print(
                                f"      -> Estado: "
                                f"{estado}"
                            )

                            print(
                                f"      -> Otras fuentes: "
                                f"{nueva_fuente}"
                            )

                        except Exception as e:

                            total_errores += 1

                            print(
                                f"    [ERROR QUITANDO "
                                f"ANDALUCÍA] "
                                f"ID {rec_id}: {e}"
                            )

                    continue

                # =================================================
                # 11. LICITACIÓN SIGUE ABIERTA
                # =================================================

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

            except requests.exceptions.RequestException as e:

                total_errores += 1

                print(
                    f"    [ERROR REQUEST] "
                    f"{titulo[:60]}... "
                    f"-> {e}"
                )

                continue

            except Exception as e:

                total_errores += 1

                print(
                    f"    [ERROR PROCESANDO] "
                    f"{titulo[:60]}... "
                    f"-> {e}"
                )

                continue

        time.sleep(0.2)

    # ====================================================
    # 12. RESUMEN FINAL
    # ====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "AUDITORÍA DE ANDALUCÍA FINALIZADA"
    )

    print(
        "=" * 60
    )

    print(
        f"Total auditadas:              "
        f"{total_registros}"
    )

    print(
        f"Eliminadas:                   "
        f"{total_eliminadas}"
    )

    print(
        f"Andalucía quitada como fuente:  "
        f"{total_fuente_quitada}"
    )

    print(
        f"Fechas actualizadas:          "
        f"{total_actualizadas}"
    )

    print(
        f"Sin cambios:                  "
        f"{total_sin_cambios}"
    )

    print(
        f"Errores:                      "
        f"{total_errores}"
    )

    print(
        "=" * 60
    )


# ========================================================
# EJECUCIÓN
# ========================================================

if __name__ == "__main__":

    auditar_licitaciones_html_andalucia()
