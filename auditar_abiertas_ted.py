from datetime import datetime, date
import os
import requests
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client
import time

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small) para auditoría TED...")
encoder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

MAPEO_NUTS_TED = {
    "ES111": "A Coruña", "ES112": "Lugo", "ES113": "Ourense", "ES114": "Pontevedra",
    "ES120": "Asturias", "ES130": "Cantabria",
    "ES211": "Álava/Araba", "ES212": "Gipuzkoa", "ES213": "Bizkaia",
    "ES220": "La Rioja", "ES230": "Navarra",
    "ES241": "Huesca", "ES242": "Teruel", "ES243": "Zaragoza",
    "ES300": "Madrid",
    "ES411": "Ávila", "ES412": "Burgos", "ES413": "León", "ES414": "Palencia",
    "ES415": "Salamanca", "ES416": "Segovia", "ES417": "Soria", "ES418": "Valladolid", "ES419": "Zamora",
    "ES421": "Albacete", "ES422": "Ciudad Real", "ES423": "Cuenca", "ES424": "Guadalajara", "ES425": "Toledo",
    "ES431": "Badajoz", "ES432": "Cáceres",
    "ES511": "Barcelona", "ES512": "Girona", "ES513": "Lleida", "ES514": "Tarragona",
    "ES521": "Alicante/Alacant", "ES522": "Castellón/Castelló", "ES523": "Valencia/València",
    "ES531": "Eivissa y Formentera", "ES532": "Mallorca", "ES533": "Menorca",
    "ES611": "Almería", "ES612": "Cádiz", "ES613": "Córdoba", "ES614": "Granada", 
    "ES615": "Huelva", "ES616": "Jaén", "ES617": "Málaga", "ES618": "Sevilla",
    "ES620": "Murcia",
    "ES630": "Ceuta",
    "ES640": "Melilla",
    "ES703": "El Hierro", "ES704": "Fuerteventura", "ES705": "Gran Canaria", 
    "ES706": "La Gomera", "ES707": "La Palma", "ES708": "Lanzarote", "ES709": "Tenerife",
    "ES1": "Noroeste (España)", "ES2": "Noreste (España)", "ES3": "Comunidad de Madrid (España)",
    "ES4": "Centro (España)", "ES5": "Este (España)", "ES6": "Sur (España)", "ES7": "Canarias (España)"
}

TAMANO_LOTE = 10  # Lotes de 10 en 10 para agilizar y evitar bloqueos

def mapear_lugar(lugar_str):
    if not lugar_str or lugar_str == "No especificado":
        return lugar_str
    elementos = [e.strip() for e in lugar_str.split(",")]
    elementos_mapeados = [MAPEO_NUTS_TED.get(el, el) for el in elementos]
    return ", ".join(dict.fromkeys(elementos_mapeados))

def procesar_campo(campo, es_lista=False):
    if isinstance(campo, list):
        limpios = [str(x) for x in campo if x]
        if es_lista:
            return list(dict.fromkeys(limpios))
        return limpios[0] if limpios else "No especificado"
    elif isinstance(campo, dict):
        return campo.get("eng") or next(iter(campo.values()), "No especificado")
    return str(campo) if campo else "No especificado"

def auditar_licitaciones_abiertas_ted():
    hoy = date.today()
    url_api = "https://api.ted.europa.eu/v3/notices/search"
    fields_solicitados = [
        "publication-number", "contract-title", "notice-title",
        "organisation-name-buyer", "deadline-receipt-request", 
        "place-of-performance", "classification-cpv", "description-proc",
        "total-value", "notice-type", "form-type"
    ]
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    total_eliminadas = 0
    total_actualizadas = 0
    total_sin_cambios = 0
    lote_contador = 1
    ids_procesados = []  # Control para evitar repetir registros en bucle

    print("🔍 Iniciando auditoría por lotes de licitaciones TED (Fecha 'No especificada')...\n")

    while True:
        try:
            query = (
                supabase.table("licitaciones")
                .select("*")
                .eq("fuente", "TED")
                .eq("fecha_fin", "No especificada")
                .limit(TAMANO_LOTE)
            )
            
            # Excluimos los IDs ya procesados en esta ejecución
            if ids_procesados:
                query = query.not_.in_("id", ids_procesados)

            response = query.execute()
            registros = response.data
        except Exception as e:
            print(f"❌ Error al consultar Supabase: {e}")
            break

        if not registros:
            print("\n🎉 ¡Proceso finalizado! No quedan más licitaciones TED pendientes de auditar.")
            break

        print(f"\n📦 --- Procesando Lote TED {lote_contador} ({len(registros)} registros) ---")
        ids_a_borrar = []

        for reg in registros:
            rec_id = reg.get("id")
            ids_procesados.append(rec_id)  # Registramos el ID para no volver a pedirlo
            
            enlace = reg.get("enlace", "")
            titulo = reg.get("titulo", "Sin título")

            if not enlace:
                continue

            # Extraer el número de publicación del enlace
            parts = enlace.split("/")
            pub_number = parts[-1] if parts else None
            if not pub_number:
                continue

            payload = {
                "query": f"publication-number = '{pub_number}'",
                "fields": fields_solicitados,
                "limit": 1
            }

            try:
                resp = requests.post(url_api, json=payload, headers=headers, timeout=15)
                if resp.status_code != 200:
                    print(f"   ⚠️ Error API TED HTTP {resp.status_code} para {pub_number}")
                    continue

                data = resp.json()
                notices = data.get("notices", [])
                
                if not notices:
                    ids_a_borrar.append(rec_id)
                    print(f"   🗑️ [A BORRAR - Ya no está en TED]: {titulo[:50]}...")
                    total_eliminadas += 1
                    continue

                aviso = notices[0]
                n_type = str(aviso.get("notice-type", "")).lower()
                f_type = str(aviso.get("form-type", "")).lower()

                # Comprobar si se ha adjudicado o cerrado
                es_adjudicado = n_type.startswith("can-") or "award" in n_type or f_type == "result"
                es_veat = n_type.startswith("dir-awa-pre") or "dir-awa-pre" in n_type or "veat" in n_type

                if es_adjudicado or es_veat:
                    ids_a_borrar.append(rec_id)
                    print(f"   🗑️ [A BORRAR - Adjudicada/Cerrada en TED]: {titulo[:50]}...")
                    total_eliminadas += 1
                    continue

                # Comprobar si ya se ha fijado una fecha de fin
                fechas_cierre = procesar_campo(aviso.get("deadline-receipt-request"), es_lista=True)
                nueva_fecha_fin = "No especificada"
                if fechas_cierre != "No especificado":
                    limite_str = fechas_cierre[0] if isinstance(fechas_cierre, list) else str(fechas_cierre)
                    nueva_fecha_fin = limite_str[:10]

                # Verificar si ha caducado
                if nueva_fecha_fin != "No especificada":
                    try:
                        f_fin_date = datetime.strptime(nueva_fecha_fin, "%Y-%m-%d").date()
                        if f_fin_date < hoy:
                            ids_a_borrar.append(rec_id)
                            print(f"   🗑️ [A BORRAR - Caducada]: {titulo[:50]}...")
                            total_eliminadas += 1
                            continue
                    except ValueError:
                        pass

                # Si la fecha fin cambió de 'No especificada' a una fecha válida, la actualizamos
                if nueva_fecha_fin != reg.get("fecha_fin"):
                    reg["fecha_fin"] = nueva_fecha_fin
                    reg["es_actualizada"] = True
                    
                    texto_completo = reg.get("texto_completo", "")
                    reg["embedding"] = encoder.encode(texto_completo).tolist()

                    supabase.table("licitaciones").upsert(reg, on_conflict="enlace").execute()
                    print(f"   🔄 [ACTUALIZADA Fecha Fin TED a {nueva_fecha_fin}]: {titulo[:50]}...")
                    total_actualizadas += 1
                else:
                    total_sin_cambios += 1

                time.sleep(0.2)

            except Exception as e:
                print(f"   ⚠️ Error procesando TED {pub_number}: {e}")
                continue

        # Borrado en bloque por lotes usando los IDs recolectados
        if ids_a_borrar:
            try:
                supabase.table("licitaciones").delete().in_("id", ids_a_borrar).execute()
                print(f"   🗑️ -> ¡{len(ids_a_borrar)} licitaciones TED eliminadas de Supabase en este lote!")
            except Exception as e:
                print(f"   ❌ Error al eliminar lote en Supabase: {e}")

        lote_contador += 1
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print("📊 RESUMEN FINAL AUDITORÍA TED:")
    print(f"  - Eliminadas (adjudicadas/caducadas/borradas): {total_eliminadas}")
    print(f"  - Actualizadas (con nueva fecha): {total_actualizadas}")
    print(f"  - Sin cambios: {total_sin_cambios}")
    print("✅ ¡Auditoría de licitaciones TED completada con éxito y sin bloqueos!")

if __name__ == "__main__":
    auditar_licitaciones_abiertas_ted()
