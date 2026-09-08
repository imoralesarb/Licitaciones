# -*- coding: utf-8 -*-
from datetime import datetime, date
import os
import time
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
TAMANO_LOTE = 50

def auditar_licitaciones_html_euskadi():
    hoy = date.today()
    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    lote_contador = 1

    print("Iniciando auditoría mediante scraping HTML de Euskadi...\n")

    while True:
        try:
            response = (
                supabase.table("licitaciones")
                .select("*")
                .eq("fuente", "Euskadi")
                .eq("fecha_fin", "No especificada")
                .limit(TAMANO_LOTE)
                .execute()
            )
            registros = response.data
        except Exception as e:
            print(f"Error al consultar Supabase: {e}")
            break

        if not registros:
            print("\n¡Proceso finalizado! No quedan más registros pendientes.")
            break

        print(f"\n--- Procesando Lote {lote_contador} ({len(registros)} registros) ---")
        ids_a_borrar = []

        for reg in registros:
            rec_id = reg.get("id")
            enlace = reg.get("enlace", "")
            titulo = reg.get("titulo", "Sin título")

            if not enlace:
                ids_a_borrar.append(rec_id)
                print(f"    [A BORRAR - Sin enlace]: {titulo[:40]}... (ID: {rec_id})")
                total_eliminadas += 1
                continue

            try:
                # Hacemos la petición HTTP a la URL del detalle
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                resp = requests.get(enlace, headers=headers, timeout=10)

                if resp.status_code != 200:
                    ids_a_borrar.append(rec_id)
                    print(f"    [A BORRAR - HTTP {resp.status_code}]")
                    print(f"      -> Título: {titulo[:50]}...")
                    print(f"      -> Enlace: {enlace}")
                    total_eliminadas += 1
                    continue

                # Parseamos el HTML con BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Buscamos los bloques de texto o etiquetas <dd> que contienen los estados y plazos
                texto_pagina = soup.get_text(separator=" ", strip=True).lower()

                # Criterio de borrado: Si el texto indica que ya no está abierto o está anulado/adjudicado
                if "anulado" in texto_pagina and "plazo de presentación" not in texto_pagina:
                    ids_a_borrar.append(rec_id)
                    print(f"    [A BORRAR - Anulado/Cerrado en HTML]")
                    print(f"      -> Título: {titulo[:50]}...")
                    print(f"      -> Enlace: {enlace}")
                    total_eliminadas += 1
                    continue

                # Extraer fechas o estados específicos buscando en las etiquetas <dd>
                dds = soup.find_all("dd")
                estado_encontrado = ""
                for dd in dds:
                    txt = dd.get_text(strip=True)
                    # Identificar si el texto contiene estados de cierre o plazos
                    if "abierto / plazo de presentación" in txt.lower() or "cerrado" in txt.lower():
                        estado_encontrado = txt

                # Si detectamos explícitamente que está cerrado
                if "cerrado" in estado_encontrado.lower():
                    ids_a_borrar.append(rec_id)
                    print(f"    [A BORRAR - Plazo cerrado según HTML]")
                    print(f"      -> Título: {titulo[:50]}...")
                    print(f"      -> Enlace: {enlace}")
                    total_eliminadas += 1
                    continue

                # Actualizamos el estado de control para evitar bucle infinito si sigue sin fecha
                supabase.table("licitaciones").update({"es_actualizada": False}).eq("id", rec_id).execute()
                total_sin_cambios += 1

                time.sleep(0.2)

            except Exception as e:
                print(f"    Error procesando enlace {enlace}: {e}")
                continue

        if ids_a_borrar:
            try:
                for i in range(0, len(ids_a_borrar), 50):
                    lote_ids = ids_a_borrar[i:i+50]
                    supabase.table("licitaciones").delete().in_("id", lote_ids).execute()
                print(f"    -> ¡{len(ids_a_borrar)} licitaciones eliminadas de Supabase!")
            except Exception as e:
                print(f"    Error al eliminar en Supabase: {e}")

        lote_contador += 1
        time.sleep(0.2)

    print("\n" + "=" * 50)
    print(f"Eliminadas: {total_eliminadas} | Sin cambios: {total_sin_cambios}")

if __name__ == "__main__":
    auditar_licitaciones_html_euskadi()
