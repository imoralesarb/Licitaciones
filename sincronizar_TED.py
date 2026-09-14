# -*- coding: utf-8 -*-
from datetime import datetime, date, timedelta
import os
import time
import re
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import requests

# ============================================================
# CONFIGURACIÓN
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Faltan SUPABASE_URL o SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
TED_URL = "https://api.ted.europa.eu/v3/notices/search"
HOY = date.today()
FECHA_DESDE = HOY - timedelta(days=2)
print("Cargando modelo de IA...")
modelo = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu"
)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def limpiar_titulo(titulo):
    """
    Limpia prefijos habituales de los títulos de TED.
    """
    if not titulo:
        return ""
    titulo = str(titulo).strip()
    separadores = [" – ", " - ", " — "]
    for separador in separadores:
        if separador in titulo:
            partes = titulo.split(separador, 1)
            if len(partes) == 2:
                titulo = partes[1].strip()
                break
    return titulo

def limpiar_organo(organo):
    """
    Limpia el nombre del órgano para las comparaciones.
    """
    if not organo:
        return ""
    return re.sub(r"\s+", " ", str(organo).strip())

def extraer_texto_multilingue(valor, idioma_preferido="spa"):
    """
    Extrae texto plano de estructuras multilingües o anidadas de la API de TED.
    """
    if valor is None:
        return None

    if isinstance(valor, dict):
        if idioma_preferido in valor and valor[idioma_preferido]:
            resultado = extraer_texto_multilingue(valor[idioma_preferido], idioma_preferido)
            if resultado:
                return resultado
        for contenido in valor.values():
            resultado = extraer_texto_multilingue(contenido, idioma_preferido)
            if resultado:
                return resultado
        return None

    if isinstance(valor, (list, tuple)):
        for elemento in valor:
            resultado = extraer_texto_multilingue(elemento, idioma_preferido)
            if resultado:
                return resultado
        return None

    texto = str(valor).strip()
    return texto if texto else None


PATRON_FECHA_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")


def extraer_fecha_iso(valor):
    """
    Extraera "YYYY-MM-DD" robustamente de cualquier formato de fecha de TED.
    """
    if valor is None:
        return None

    if isinstance(valor, (list, tuple)):
        for elemento in valor:
            resultado = extraer_fecha_iso(elemento)
            if resultado:
                return resultado
        return None

    if isinstance(valor, dict):
        for contenido in valor.values():
            resultado = extraer_fecha_iso(contenido)
            if resultado:
                return resultado
        return None

    coincidencia = PATRON_FECHA_ISO.search(str(valor))
    if not coincidencia:
        return None

    fecha_str = coincidencia.group(1)
    try:
        date.fromisoformat(fecha_str)
    except ValueError:
        return None

    return fecha_str


def normalizar_valor(valor):
    """
    Normaliza valores para poder comparar correctamente.
    """
    if valor is None:
        return None
    if isinstance(valor, float):
        return round(valor, 2)
    return valor

def obtener_importe(notice):
    """
    Obtiene el importe total del aviso TED (devuelve 0.00 si no existe).
    """
    valor = notice.get("total-value")
    if valor is None:
        valor = notice.get("total-value-cur")

    if isinstance(valor, (list, tuple)):
        valor = valor[0] if valor else None
    if isinstance(valor, dict):
        valor = valor.get("value") or valor.get("amount") or next(iter(valor.values()), None)

    if valor is None:
        return 0.00
    try:
        return float(valor)
    except (ValueError, TypeError):
        return 0.00

def obtener_tipo_contrato(notice):
    """
    Traduce el tipo de contrato de TED.
    """
    naturaleza = str(notice.get("contract-nature", "")).strip().lower()
    if naturaleza == "supplies":
        return "Suministros"
    if naturaleza == "services":
        return "Servicios"
    if naturaleza == "works":
        return "Obras"
    return None

def obtener_lugar(place):
    """
    Convierte información de NUTS de TED a un lugar legible.
    """
    mapa_nuts = {
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
        "ES4": "Centro (España)", "ES5": "Este (España)", "ES6": "Sur (España)", "ES7": "Canarias (España)",
        "ES": "España", "ESP": "España",
    }

    def _mapear_codigo(codigo):
        codigo = str(codigo).strip()
        return mapa_nuts.get(codigo.upper(), codigo)

    if not place:
        return None

    if isinstance(place, dict):
        for clave in ["nuts", "nuts-code", "nutsCode", "code"]:
            valor = place.get(clave)
            if valor:
                if isinstance(valor, list):
                    valor = valor[0]
                return _mapear_codigo(valor)

    if isinstance(place, list):
        lugares = []
        for elemento in place:
            lugar = obtener_lugar(elemento)
            if lugar:
                lugares.append(lugar)
        if lugares:
            return ", ".join(dict.fromkeys(lugares))

    texto = str(place).strip()
    if not texto:
        return None

    codigos = [c.strip() for c in texto.split(",") if c.strip()]
    if len(codigos) > 1:
        nombres_unicos = list(dict.fromkeys(_mapear_codigo(c) for c in codigos))
        MAX_LUGARES_MOSTRADOS = 3
        if len(nombres_unicos) > MAX_LUGARES_MOSTRADOS:
            resto = len(nombres_unicos) - MAX_LUGARES_MOSTRADOS
            return ", ".join(nombres_unicos[:MAX_LUGARES_MOSTRADOS]) + f" y {resto} más"
        return ", ".join(nombres_unicos)

    return _mapear_codigo(texto)

def construir_texto_embedding(elemento):
    """
    Construye el texto utilizado para el embedding.
    """
    partes = [
        elemento.get("titulo", ""),
        elemento.get("organo", ""),
        elemento.get("tipo_contrato", ""),
        elemento.get("lugar_ejecucion", ""),
        elemento.get("cpv", ""),
        elemento.get("texto_completo", "")
    ]
    partes_limpias = [str(p).strip() for p in partes if p is not None and str(p).strip()]
    return " ".join(partes_limpias)

def es_resultado_o_adjudicado(notice):
    n_type = str(notice.get("notice-type", "")).strip().lower()
    f_type = str(notice.get("form-type", "")).strip().lower()
    return n_type.startswith("can-") or "award" in n_type or f_type == "result"

def es_veat(notice):
    n_type = str(notice.get("notice-type", "")).strip().lower()
    return n_type.startswith("dir-awa-pre") or "dir-awa-pre" in n_type or "veat" in n_type

# ============================================================
# DESCARGAR TED
# ============================================================
def descargar_avisos_ted():
    fecha_inicio = FECHA_DESDE.strftime("%Y%m%d")
    fecha_fin_str = HOY.strftime("%Y%m%d")

    print(f"Consultando la API de TED para España ({fecha_inicio} a {fecha_fin_str})...")

    campos_solicitados = [
        "publication-number",
        "contract-title",
        "notice-title",
        "organisation-name-buyer",
        "publication-date",
        "deadline-receipt-request",
        "place-of-performance",
        "classification-cpv",
        "description-proc",
        "total-value",
        "total-value-cur",
        "notice-type",
        "form-type",
        "contract-nature"  # <-- ASEGÚRATE DE INCLUIR ESTE CAMPO AQUÍ
    ]

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    limit = 250

    avisos = []
    iteration_next_token = None

    while True:
        payload = {
            "query": f"publication-date >= '{fecha_inicio}' AND publication-date <= '{fecha_fin_str}' AND buyer-country = 'ESP'",
            "fields": campos_solicitados,
            "paginationMode": "ITERATION",
            "limit": limit
        }
        if iteration_next_token:
            payload["iterationNextToken"] = iteration_next_token

        try:
            respuesta = requests.post(TED_URL, json=payload, headers=headers, timeout=30)
        except Exception as e:
            print(f"Excepción conectando con TED: {e}")
            break

        if respuesta.status_code == 200:
            datos = respuesta.json()
            resultados = datos.get("notices", [])
            if not resultados:
                break

            avisos.extend(resultados)
            print(f"  Lote TED descargado: {len(resultados)} registros (total {len(avisos)})")

            iteration_next_token = datos.get("iterationNextToken")
            if not iteration_next_token or len(resultados) < limit:
                break
            time.sleep(0.3)

        elif respuesta.status_code == 429:
            print("Límite de peticiones TED alcanzado (429). Esperando 5 segundos...")
            time.sleep(5)
            continue

        else:
            print(f"Error API TED HTTP {respuesta.status_code}: {respuesta.text}")
            break

    print(f"Total avisos TED descargados: {len(avisos)}")
    return avisos


avisos_ted = descargar_avisos_ted()

# ============================================================
# CARGAR REGISTROS EXISTENTES
# ============================================================
print("Cargando registros existentes desde Supabase...")
existentes_resp = supabase.table("licitaciones").select(
    "id,enlace,titulo,organo,fecha,importe,fuente,fecha_fin,tipo_contrato,cpv,lugar_ejecucion,texto_completo,es_novedad,es_actualizada"
).execute()
registros_existentes = existentes_resp.data or []
print(f"Registros totales cargados desde Supabase: {len(registros_existentes)}")

# ============================================================
# MAPAS Y RESTABLECIMIENTO DE BANDERAS
# ============================================================
mapa_enlaces = {}
mapa_claves = {}
for registro in registros_existentes:
    enlace = str(registro.get("enlace", "")).strip()
    if enlace:
        mapa_enlaces[enlace] = registro
    titulo = limpiar_titulo(registro.get("titulo", "")).casefold()
    organo = limpiar_organo(registro.get("organo", "")).casefold()
    if titulo and organo:
        mapa_claves[(titulo, organo)] = registro

ids_flags_ted = []
for item in registros_existentes:
    fuente_item = str(item.get("fuente", ""))
    if "ted" in fuente_item.casefold() and (item.get("es_novedad") is True or item.get("es_actualizada") is True):
        ids_flags_ted.append(item["id"])

print(f"Reseteando etiquetas anteriores de {len(ids_flags_ted)} registros de TED...")
BATCH_RESET = 25
for inicio in range(0, len(ids_flags_ted), BATCH_RESET):
    lote_ids = ids_flags_ted[inicio:inicio + BATCH_RESET]
    for intento in range(1, 4):
        try:
            supabase.table("licitaciones").update({
                "es_novedad": False,
                "es_actualizada": False
            }).in_("id", lote_ids).execute()
            break
        except Exception:
            if intento < 3:
                time.sleep(2 * intento)

# ============================================================
# PROCESAMIENTO
# ============================================================
nuevas = 0
actualizadas = 0
sin_cambios = 0
duplicados = 0
resultados_filtrados = 0
veat = 0
caducados = 0
licitaciones_nuevas = []

for notice in avisos_ted:
    if es_resultado_o_adjudicado(notice):
        resultados_filtrados += 1
        continue
    if es_veat(notice):
        veat += 1
        continue

    publication_number = extraer_texto_multilingue(notice.get("publication-number")) or ""
    if not publication_number:
        continue
    enlace = f"https://ted.europa.eu/es/notice/-/detail/{publication_number}"

    titulo_original = (
        extraer_texto_multilingue(notice.get("contract-title"))
        or extraer_texto_multilingue(notice.get("notice-title"))
        or ""
    )
    titulo_limpio = limpiar_titulo(titulo_original)

    organo_bruto = extraer_texto_multilingue(notice.get("organisation-name-buyer"))
    organo = limpiar_organo(organo_bruto)

    fecha_str = extraer_fecha_iso(notice.get("publication-date")) or ""
    fecha_fin_str = extraer_fecha_iso(notice.get("deadline-receipt-request"))

    importe = obtener_importe(notice)
    tipo_contrato = obtener_tipo_contrato(notice)
    lugar_ejecucion = obtener_lugar(notice.get("place-of-performance"))
    
    cpv = notice.get("classification-cpv")
    if isinstance(cpv, list):
        cpv = ", ".join(str(x) for x in cpv)
    if cpv is not None:
        cpv = str(cpv).strip()

    texto_completo = notice.get("description-proc")
    if texto_completo is not None:
        texto_completo = str(texto_completo).strip()

    if fecha_fin_str:
        try:
            fecha_fin_date = date.fromisoformat(fecha_fin_str)
            if fecha_fin_date < HOY:
                caducados += 1
                continue
        except ValueError:
            pass

    reg_existente = mapa_enlaces.get(enlace)
    if reg_existente is None:
        clave = (titulo_limpio.casefold(), limpiar_organo(organo).casefold())
        reg_existente = mapa_claves.get(clave)

    if reg_existente is not None:
        cambios = {}
        if reg_existente.get("titulo") != titulo_limpio:
            cambios["titulo"] = titulo_limpio
        importe_existente = normalizar_valor(reg_existente.get("importe"))
        importe_nuevo = normalizar_valor(importe)
        if importe_existente != importe_nuevo:
            cambios["importe"] = importe
        fecha_fin_existente = reg_existente.get("fecha_fin")
        if fecha_fin_existente != fecha_fin_str:
            cambios["fecha_fin"] = fecha_fin_str
        if not reg_existente.get("tipo_contrato") and tipo_contrato:
            cambios["tipo_contrato"] = tipo_contrato
            
        fuente_existente = str(reg_existente.get("fuente", "")).strip()
        fuentes = [f.strip() for f in fuente_existente.split(",") if f.strip()]
        if not any(f.casefold() == "ted" for f in fuentes):
            fuentes.append("TED")
            cambios["fuente"] = ", ".join(dict.fromkeys(fuentes))

        if "titulo" in cambios or "importe" in cambios or "fecha_fin" in cambios or "tipo_contrato" in cambios:
            cambios["es_actualizada"] = True
            try:
                supabase.table("licitaciones").update(cambios).eq("id", reg_existente["id"]).execute()
                actualizadas += 1
            except Exception as e:
                print(f"Error actualizando {reg_existente['id']}: {e}")
        elif cambios:
            try:
                supabase.table("licitaciones").update(cambios).eq("id", reg_existente["id"]).execute()
            except Exception:
                pass
        else:
            sin_cambios += 1
        continue

    elemento = {
        "enlace": enlace,
        "titulo": titulo_limpio,
        "organo": organo,
        "fuente": "TED",
        "fecha": fecha_str,
        "importe": importe,
        "tipo_contrato": tipo_contrato,
        "cpv": cpv,
        "fecha_fin": fecha_fin_str,
        "lugar_ejecucion": lugar_ejecucion,
        "texto_completo": texto_completo,
        "es_novedad": True,
        "es_actualizada": False
    }
    
    texto_embedding = construir_texto_embedding(elemento)
    try:
        embedding = modelo.encode(texto_embedding).tolist()
        elemento["embedding"] = embedding
    except Exception:
        elemento["embedding"] = None
        
    licitaciones_nuevas.append(elemento)
    nuevas += 1

# ============================================================
# INSERCIÓN Y LIMPIEZA
# ============================================================
if licitaciones_nuevas:
    print(f"Insertando {len(licitaciones_nuevas)} licitaciones nuevas...")
    BATCH_INSERT = 5
    for inicio in range(0, len(licitaciones_nuevas), BATCH_INSERT):
        lote = licitaciones_nuevas[inicio:inicio + BATCH_INSERT]
        try:
            supabase.table("licitaciones").insert(lote).execute()
        except Exception as e:
            print(f"Error insertando lote: {e}")

print("Sincronización TED finalizada.")
