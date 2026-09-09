# -*- coding: utf-8 -*-
import os
import requests
from supabase import create_client, Client

# ============================================================
# CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def auditar_licitaciones_cerradas_navarra():
    resource_id = "dda1af7c-0dcd-4992-9852-ded6b1e7625d"
    url_api = f"https://datosabiertos.navarra.es/es/api/3/action/datastore_search?resource_id={resource_id}&limit=1000"
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    print("Consultando la API de Contratación de Navarra para auditoría de cierre...")
    try:
        response = requests.get(url_api, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Error HTTP {response.status_code}")
            return
        data = response.json()
        results = data.get("result", {}).get("records", [])
    except Exception as e:
        print(f"Error conectando con la API de Navarra: {e}")
        return

    print(f"Total registros vigentes obtenidos de la API: {len(results)}")

    # Construir conjunto de claves activas basadas en (organo, titulo)
    claves_en_api = set()
    for aviso in results:
        titulo = str(aviso.get("BreveDescripcion") or "Sin descripción").strip()
        organo = str(aviso.get("Organo") or "No especificado").strip()
        entidad = str(aviso.get("Entidad") or "No especificada").strip()
        organo_completo = f"{entidad} - {organo}" if entidad else organo
        claves_en_api.add((organo_completo, titulo))

    # Obtener registros de Supabase con fuente Contratación Navarra
    print("Obteniendo registros de Supabase para comparar...")
    try:
        response_db = (
            supabase.table("licitaciones")
            .select("id, titulo, organo")
            .eq("fuente", "Contratación Navarra")
            .execute()
        )
        registros_db = response_db.data
    except Exception as e:
        print(f"Error al consultar Supabase: {e}")
        return

    print(f"Total registros en Supabase con fuente 'Contratación Navarra': {len(registros_db)}")

    ids_a_borrar = []
    total_eliminadas = 0

    for reg in registros_db:
        rec_id = reg.get("id")
        titulo = reg.get("titulo", "")
        organo = reg.get("organo", "")

        clave = (organo, titulo)
        if clave not in claves_en_api:
            ids_a_borrar.append(rec_id)
            print(f"   [A BORRAR - Ya no está en la API de Navarra]: {titulo[:50]}...")
            total_eliminadas += 1

    # Borrado en lotes de 50 para optimizar llamadas a Supabase
    if ids_a_borrar:
        print(f"\nEliminando {len(ids_a_borrar)} licitaciones cerradas de Supabase...")
        for i in range(0, len(ids_a_borrar), 50):
            lote_ids = ids_a_borrar[i : i + 50]
            try:
                supabase.table("licitaciones").delete().in_("id", lote_ids).execute()
            except Exception as e:
                print(f"Error al eliminar lote en Supabase: {e}")
        print("¡Eliminación completada con éxito!")
    else:
        print("\nNo hay licitaciones para eliminar. Todas siguen vigentes en la API.")

    print("\n" + "=" * 50)
    print(" RESUMEN FINAL AUDITORÍA CONTRATACIÓN NAVARRA:")
    print(f"  - Eliminadas (cerradas/fuera de API): {total_eliminadas}")
    print(" ¡Auditoría de cierre completada con éxito!")

if __name__ == "__main__":
    auditar_licitaciones_cerradas_navarra()
