# -*- coding: utf-8 -*-
from datetime import datetime, date
import os
import requests
from supabase import create_client, Client

# ============================================================
# CONFIGURACIÓN
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "TU_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "TU_SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def auditar_licitaciones_rioja():
    url_json = "https://ias1.larioja.org/opendata/download?r=Y2Q9MTc5fGNmPTA0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01"
    }

    print("📥 Descargando el archivo JSON oficial del Gobierno de La Rioja...")
    try:
        response = requests.get(url_json, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"❌ Error HTTP {response.status_code} al descargar los datos.")
            return
        data_json = response.json()
        results = data_json.get("data", [])
    except Exception as e:
        print(f"❌ Error conectando con el portal de La Rioja: {e}")
        return

    print(f"Total registros obtenidos del JSON oficial: {len(results)}")

    # Mapear los datos actuales del JSON usando (titulo_normalizado, organo_normalizado) como clave
    estados_excluidos = {"resuelta", "cerrada", "adjudicada", "anulada", "desistida", "renunciada"}
    json_datos_map = {}

    for aviso in results:
        t_str = str(aviso.get("OBJETO_CONTRATO", "")).strip().strip('"').lower()
        o_str = str(aviso.get("ORGANO_CONTRATACION", "")).strip().lower()
        
        if not t_str or not o_str:
            continue

        # Estado de la licitación en origen
        estado_licitacion = str(aviso.get("ESTADO_LICITACION", "")).strip().lower()
        esta_abierta = not any(est in estado_licitacion for est in estados_excluidos)

        # Extraer fecha fin de ofertas si existe
        f_fin_str = "No especificada"
        f_fin_raw = aviso.get("FECHA_HORA_FIN_OFERTAS")
        if f_fin_raw:
            try:
                f_fin_str = str(f_fin_raw)[:10].replace("/", "-")
            except Exception:
                pass

        json_datos_map[(t_str, o_str)] = {
            "fecha_fin": f_fin_str,
            "esta_abierta": esta_abierta,
            "estado_origen": estado_licitacion
        }

    # 2. Obtener de Supabase las licitaciones correspondientes a la fuente de La Rioja
    print("🔍 Consultando licitaciones de La Rioja en Supabase...")
    try:
        resp = supabase.table("licitaciones").select("id, titulo, organo, fecha_fin, fuente").ilike("fuente", "%Gobierno de La Rioja%").execute()
        registros_db = resp.data
    except Exception as e:
        print(f"❌ Error al consultar Supabase: {e}")
        return

    print(f"Registros encontrados en Supabase para auditar: {len(registros_db)}")

    hoy_date = datetime.now().date()
    eliminadas_caducadas = 0
    actualizadas_fecha = 0
    eliminadas_cerradas = 0

    for reg in registros_db:
        reg_id = reg["id"]
        t_db = str(reg.get("titulo", "")).strip().lower()
        o_db = str(reg.get("organo", "")).strip().lower()
        fecha_fin_db = str(reg.get("fecha_fin", "No especificada")).strip()

        # REGLA 1: Si la fecha fin ya está caducada en Supabase, eliminarla directamente
        if fecha_fin_db != "No especificada":
            try:
                f_fin_date = datetime.strptime(fecha_fin_db[:10], "%Y-%m-%d").date()
                if f_fin_date < hoy_date:
                    supabase.table("licitaciones").delete().eq("id", reg_id).execute()
                    eliminadas_caducadas += 1
                    continue
            except Exception:
                pass

        clave = (t_db, o_db)
        info_json = json_datos_map.get(clave)

        # Si ya no aparece en el JSON oficial, se salta o se evalúa si conviene eliminar
        if not info_json:
            continue

        # REGLA 2: Si en Supabase estaba como "No especificada", comprobar si el JSON ahora trae fecha válida
        if fecha_fin_db == "No especificada" or not fecha_fin_db:
            nueva_fecha_fin = info_json["fecha_fin"]
            if nueva_fecha_fin != "No especificada":
                try:
                    f_nueva_date = datetime.strptime(nueva_fecha_fin[:10], "%Y-%m-%d").date()
                    # Si la nueva fecha es válida y futura (o de hoy), la actualizamos
                    if f_nueva_date >= hoy_date:
                        supabase.table("licitaciones").update({"fecha_fin": nueva_fecha_fin, "es_actualizada": True}).eq("id", reg_id).execute()
                        actualizadas_fecha += 1
                        continue
                    else:
                        # Si la fecha que trae el JSON ya está caducada, la eliminamos
                        supabase.table("licitaciones").delete().eq("id", reg_id).execute()
                        eliminadas_caducadas += 1
                        continue
                except Exception:
                    pass

        # REGLA 3: Si sigue sin fecha válida, comprobar si el estado en origen sigue abierto
        # Si ya no está abierta en el portal de La Rioja, eliminarla de Supabase
        if not info_json["esta_abierta"]:
            supabase.table("licitaciones").delete().eq("id", reg_id).execute()
            eliminadas_cerradas += 1

    print("\n========================================")
    print("📊 RESUMEN DE LA AUDITORÍA DE LA RIOJA:")
    print(f"🗑️ Eliminadas por fecha caducada: {eliminadas_caducadas}")
    print(f"📅 Actualizadas con nueva fecha de cierre válida: {actualizadas_fecha}")
    print(f"❌ Eliminadas por haber cerrado su estado en origen: {eliminadas_cerradas}")
    print("========================================")

if __name__ == "__main__":
    auditar_licitaciones_rioja()
