from datetime import datetime, date, timedelta
import os
import time
import requests
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Cargando modelo de IA (multilingual-e5-small) para TED...")
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

# ============================================================
# 2. FUNCIONES AUXILIARES Y DE LIMPIEZA
# ============================================================

def mapear_lugar(lugar_str):
    if not lugar_str or lugar_str == "No especificado":
        return lugar_str
    elementos = [e.strip() for e in lugar_str.split(",")]
    elementos_mapeados = [MAPEO_NUTS_TED.get(el, el) for el in elementos]
    return ", ".join(dict.fromkeys(elementos_mapeados))

def limpiar_titulo(titulo):
    if isinstance(titulo, list):
        titulo = " ".join([str(x) for x in titulo if x])
    elif not isinstance(titulo, str):
        titulo = str(titulo) if titulo else ""
    titulo = titulo.strip()
    for separador in [" – ", " - ", " — "]:
        if separador in titulo:
            partes = titulo.split(separador)
            if len(partes) > 1:
                titulo = partes[-1].strip()
                break
    return titulo.strip()

def limpiar_organo(organo):
    if isinstance(organo, list):
        organo = organo[0] if organo else ""
    elif isinstance(organo, dict):
        organo = organo.get("eng") or next(iter(organo.values()), "")
    texto = str(organo).strip()
    for char in ["[", "]", "'", '"']:
        texto = texto.replace(char, "")
    return texto.strip()

def procesar_campo(campo, es_lista=False):
    if isinstance(campo, list):
        limpios = [str(x) for x in campo if x]
        if es_lista:
            return list(dict.fromkeys(limpios))
        return limpios[0] if limpios else "No especificado"
    elif isinstance(campo, dict):
        return campo.get("eng") or next(iter(campo.values()), "No especificado")
    return str(campo) if campo else "No especificado"

# ============================================================
# 3. CONSULTA A LA API DE TED (MODO ITERATION)
# ============================================================

def consultar_ted_api_scroll():
    url = "https://api.ted.europa.eu/v3/notices/search"
    fields_solicitados = [
        "publication-number", "contract-title", "notice-title",
        "organisation-name-buyer", "publication-date",
        "deadline-receipt-request", "place-of-performance",
        "classification-cpv", "description-proc",
        "total-value", "total-value-cur", "notice-type", "form-type"
    ]
    
    # Reducimos la ventana a 7 días ya que se ejecuta a diario para evitar sobrecarga
    hoy = date.today()
    fecha_inicio = (hoy - timedelta(days=7)).strftime("%Y%m%d")
    fecha_fin_str = hoy.strftime("%Y%m%d")
    
    todas_licitaciones = []
    iteration_next_token = None
    limit = 250
    
    print(f"Consultando la API de TED para España ({fecha_inicio} a {fecha_fin_str})...")
    
    while True:
        payload = {
            "query": f"publication-date >= '{fecha_inicio}' AND publication-date <= '{fecha_fin_str}' AND buyer-country = 'ESP'",
            "fields": fields_solicitados,
            "paginationMode": "ITERATION",
            "limit": limit
        }
        if iteration_next_token:
            payload["iterationNextToken"] = iteration_next_token
            
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                notices = data.get("notices", [])
                if not notices:
                    break
                todas_licitaciones.extend(notices)
                print(f"Lote TED descargado ({len(notices)} avisos). Acumulado: {len(todas_licitaciones)}")
                
                iteration_next_token = data.get("iterationNextToken")
                if not iteration_next_token or len(notices) < limit:
                    break
                time.sleep(0.3)
            elif response.status_code == 429:
                print("Límite de peticiones TED alcanzado (429). Esperando 5 segundos...")
                time.sleep(5)
                continue
            else:
                print(f"Error API TED HTTP {response.status_code}: {response.text}")
                break
        except Exception as e:
            print(f"Excepción conectando con TED: {e}")
            break
            
    return todas_licitaciones

# ============================================================
# 4. LIMPIEZA Y SINCRONIZACIÓN DE TED EN SUPABASE
# ============================================================

def sincronizar_licitaciones_ted():
    hoy = datetime.now().date()
    
    # 1. Limpieza de registros TED caducados en Supabase
    try:
        supabase.table("licitaciones").delete().eq("fuente", "TED").lt("fecha_fin", str(hoy)).neq("fecha_fin", "No especificada").execute()
        print("🧹 Licitaciones TED caducadas eliminadas de Supabase.")
    except Exception as e:
        print(f"⚠️ Aviso al limpiar caducadas de TED: {e}")

    # Obtener avisos de la API de TED
    lics_ted = consultar_ted_api_scroll()
    if not lics_ted:
        print("ℹ️ No se obtuvieron avisos de la API de TED.")
        return

    # Cargar estado actual de Supabase para evitar duplicados y detectar actualizaciones
    try:
        existentes_resp = supabase.table("licitaciones").select("enlace, titulo, organo, fecha, fuente").execute()
        mapa_enlaces = {item["enlace"]: item for item in existentes_resp.data}
        registros_existentes = {
            (limpiar_titulo(item.get("titulo", "")).lower(), limpiar_organo(item.get("organo", "")).lower())
            for item in existentes_resp.data
        }
        print(f"Registros en Supabase cargados para validación TED: {len(existentes_resp.data)}")
    except Exception as e:
        print(f"❌ Error conectando con Supabase: {e}")
        return

    licitaciones_validas = []
    filtrados_adjudicados = 0
    filtrados_caducados = 0
    filtrados_veat = 0
    filtrados_duplicados_placsp = 0

    for aviso in lics_ted:
        num = aviso.get("publication-number")
        enlace = f"https://ted.europa.eu/en/notice/-/detail/{num}"
        
        c_title = procesar_campo(aviso.get("contract-title"))
        n_title = procesar_campo(aviso.get("notice-title"))
        titulo_bruto = c_title if c_title != "No especificado" else (n_title if n_title != "No especificado" else "Sin título")
        titulo_limpio = limpiar_titulo(titulo_bruto)
        
        organo_bruto = aviso.get("organisation-name-buyer")
        organo_limpio = limpiar_organo(organo_bruto)
        
        n_type = str(aviso.get("notice-type", "")).lower()
        f_type = str(aviso.get("form-type", "")).lower()
        
        # Filtrar adjudicados o resultados
        es_adjudicado = n_type.startswith("can-") or "award" in n_type or f_type == "result"
        if es_adjudicado:
            filtrados_adjudicados += 1
            if enlace in mapa_enlaces:
                try:
                    supabase.table("licitaciones").delete().eq("enlace", enlace).execute()
                except Exception:
                    pass
            continue

        # Filtrar VEAT
        es_veat = n_type.startswith("dir-awa-pre") or "dir-awa-pre" in n_type or "veat" in n_type
        if es_veat:
            filtrados_veat += 1
            continue
            
        # Fechas de cierre / caducidad
        fechas_cierre = procesar_campo(aviso.get("deadline-receipt-request"), es_lista=True)
        fecha_fin_str = "No especificada"
        if fechas_cierre != "No especificado":
            limite_str = fechas_cierre[0] if isinstance(fechas_cierre, list) else str(fechas_cierre)
            fecha_fin_str = limite_str[:10]
            try:
                fecha_cierre_date = datetime.strptime(fecha_fin_str, "%Y-%m-%d").date()
                if fecha_cierre_date < hoy:
                    filtrados_caducados += 1
                    continue
            except ValueError:
                pass

        fecha_pub_raw = aviso.get("publication-date", "")
        fecha_str = fecha_pub_raw[:10] if fecha_pub_raw else date.today().strftime("%Y-%m-%d")
        
        importe_raw = aviso.get("total-value")
        if isinstance(importe_raw, dict):
            importe_val = importe_raw.get("value") or next(iter(importe_raw.values()), None)
        elif isinstance(importe_raw, list):
            importe_val = importe_raw[0] if importe_raw else None
        else:
            importe_val = importe_raw

        try:
            importe = float(importe_val) if importe_val is not None else 0.0
        except (ValueError, TypeError):
            importe = 0.0

        cpvs = ", ".join(procesar_campo(aviso.get("classification-cpv"), es_lista=True))
        lugares_raw = ", ".join(procesar_campo(aviso.get("place-of-performance"), es_lista=True))
        lugares = mapear_lugar(lugares_raw)
        descripcion = procesar_campo(aviso.get("description-proc"))
        
        texto_completo = f"passage: Título: {titulo_limpio}. Objeto: {descripcion}. Órgano: {organo_limpio}. CPV: {cpvs}. Lugar: {lugares}. Importe: {importe} EUR."

        clave_duplicado = (titulo_limpio.lower(), organo_limpio.lower())
        
        es_novedad = False
        es_actualizada = False

        if enlace in mapa_enlaces:
            reg_existente = mapa_enlaces[enlace]
            if fecha_str > reg_existente.get("fecha", ""):
                es_actualizada = True
        else:
            # Si el enlace no está, pero la combinación de Título y Órgano ya existe (ej. ya la subió PLACSP), la omitimos para no duplicar
            if clave_duplicado in registros_existentes:
                filtrados_duplicados_placsp += 1
                continue
            else:
                es_novedad = True

        embedding = encoder.encode(texto_completo).tolist()

        elemento = {
            "titulo": titulo_limpio,
            "organo": organo_limpio,
            "fecha": fecha_str,
            "importe": importe,
            "enlace": enlace,
            "texto_completo": texto_completo,
            "embedding": embedding,
            "fecha_fin": fecha_fin_str,
            "lugar_ejecucion": lugares,
            "cpv": cpvs,
            "es_novedad": es_novedad,
            "es_actualizada": es_actualizada,
            "fuente": "TED"
        }
        
        licitaciones_validas.append(elemento)

    print(f"\nEstadísticas TED - Duplicados evitados (ya estaban en PLACSP): {filtrados_duplicados_placsp} | Adjudicados: {filtrados_adjudicados} | VEAT: {filtrados_veat} | Caducados: {filtrados_caducados}")
    print(f"Licitaciones TED listas para sincronizar: {len(licitaciones_validas)}")

    # Inserción / actualización en Supabase por lotes
    if licitaciones_validas:
        print("🚀 Subiendo licitaciones TED a Supabase...")
        tamano_lote = 25
        for i in range(0, len(licitaciones_validas), tamano_lote):
            lote = licitaciones_validas[i:i + tamano_lote]
            try:
                supabase.table("licitaciones").upsert(lote, on_conflict="enlace").execute()
                print(f"  -> Lote TED {i // tamano_lote + 1} procesado con éxito ({len(lote)} registros).")
            except Exception as e:
                print(f"❌ Error al subir lote TED {i // tamano_lote + 1}: {e}")
        print("✅ ¡Sincronización de TED completada con éxito!")
    else:
        print("ℹ️ No hay licitaciones TED nuevas que sincronizar.")

if __name__ == "__main__":
    sincronizar_licitaciones_ted()
