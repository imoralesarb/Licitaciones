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
    Obtiene el importe total del aviso TED.
    """
    valor = notice.get("total-value")
    if valor is None:
        valor = notice.get("total-value-cur")
    if valor is None:
        return None
    try:
        return float(valor)
    except (ValueError, TypeError):
        return None

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
        "ES": "España",
        "ES11": "Galicia",
        "ES12": "Principado de Asturias",
        "ES13": "Cantabria",
        "ES21": "País Vasco",
        "ES22": "Navarra",
        "ES23": "La Rioja",
        "ES24": "Aragón",
        "ES30": "Comunidad de Madrid",
        "ES41": "Castilla y León",
        "ES42": "Castilla-La Mancha",
        "ES43": "Extremadura",
        "ES51": "Cataluña",
        "ES52": "Comunidad Valenciana",
        "ES53": "Illes Balears",
        "ES61": "Andalucía",
        "ES62": "Región de Murcia",
        "ES63": "Ciudad Autónoma de Ceuta",
        "ES64": "Ciudad Autónoma de Melilla",
        "ES70": "Canarias",
        "ES211": "Álava/Araba",
        "ES212": "Gipuzkoa",
        "ES213": "Bizkaia",
    }
    if not place:
        return None
    if isinstance(place, dict):
        for clave in ["nuts", "nuts-code", "nutsCode", "code"]:
            valor = place.get(clave)
            if valor:
                if isinstance(valor, list):
                    valor = valor[0]
                return mapa_nuts.get(str(valor).upper(), str(valor))
    if isinstance(place, list):
        lugares = []
        for elemento in place:
            lugar = obtener_lugar(elemento)
            if lugar:
                lugares.append(lugar)
        if lugares:
            return ", ".join(dict.fromkeys(lugares))
    texto = str(place).strip()
    if texto:
        return mapa_nuts.get(texto.upper(), texto)
    return None

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
    """
    Determina si el aviso corresponde a un resultado/adjudicación.
    """
    n_type = str(notice.get("notice-type", "")).strip().lower()
    f_type = str(notice.get("form-type", "")).strip().lower()
    return n_type.startswith("can-") or "award" in n_type or f_type == "result"

def es_veat(notice):
    """
    Determina si el aviso corresponde a un VEAT.
    """
    n_type = str(notice.get("notice-type", "")).strip().lower()
    return n_type.startswith("dir-awa-pre") or "dir-awa-pre" in n_type or "veat" in n_type

# ============================================================
# DESCARGAR TED
# ============================================================
print(f"Consultando la API de TED para España ({FECHA_DESDE.strftime('%Y%m%d')} a {HOY.strftime('%Y%m%d')})...")
payload = {
    "query": f"publication-date >= {FECHA_DESDE.strftime('%Y-%m-%d')} AND publication-date <= {HOY.strftime('%Y-%m-%d')} AND buyer-country = ESP",
    "fields": [
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
        "contract-nature"
    ],
    "paginationMode": "ITERATION",
    "limit": 250
}
avisos_ted = []
iteration = 1

while True:
    payload["iteration"] = iteration
    try:
        respuesta = requests.post(TED_URL, json=payload, timeout=120)
        respuesta.raise_for_status()
        datos = respuesta.json()
    except Exception as e:
        print(f"Error consultando TED: {e}")
        break
    resultados = datos.get("notices", [])
    if not resultados:
        break
    avisos_ted.extend(resultados)
    print(f"  Lote TED descargado: {len(resultados)} registros (total {len(avisos_ted)})")
    if len(resultados) < 250:
        break
    iteration += 1

print(f"Total avisos TED descargados: {len(avisos_ted)}")

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
# MAPAS DE REGISTROS EXISTENTES
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

# ============================================================
# RESETEAR ETIQUETAS ANTERIORES DE TED
# ============================================================
ids_flags_ted = []
for item in registros_existentes:
    fuente_item = str(item.get("fuente", ""))
    if "ted" in fuente_item.casefold() and (item.get("es_novedad") is True or item.get("es_actualizada") is True):
        ids_flags_ted.append(item["id"])

print(f"Reseteando etiquetas anteriores de {len(ids_flags_ted)} registros de TED...")
BATCH_RESET = 25
for inicio in range(0, len(ids_flags_ted), BATCH_RESET):
    lote_ids = ids_flags_ted[inicio:inicio + BATCH_RESET]
    actualizado = False
    for intento in range(1, 4):
        try:
            supabase.table("licitaciones").update({
                "es_novedad": False,
                "es_actualizada": False
            }).in_("id", lote_ids).execute()
            actualizado = True
            print(f"  -> Lote {inicio // BATCH_RESET + 1} actualizado correctamente.")
            break
        except Exception as e:
            print(f"  -> Intento {intento}/3 fallido para lote {inicio // BATCH_RESET + 1}: {e}")
            if intento < 3:
                time.sleep(2 * intento)
    if not actualizado:
        print(f"  -> ERROR: no se pudo actualizar el lote {inicio // BATCH_RESET + 1}")

print(f"Etiquetas anteriores reseteadas: {len(ids_flags_ted)} registros.")

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

    publication_number = str(notice.get("publication-number", "")).strip()
    if not publication_number:
        continue
    enlace = "https://ted.europa.eu/es/notice/" + publication_number
    titulo_original = notice.get("contract-title") or notice.get("notice-title") or ""
    titulo_limpio = limpiar_titulo(titulo_original)
    organo = limpiar_organo(notice.get("organisation-name-buyer", ""))
    fecha_str = str(notice.get("publication-date", "")).strip()
    fecha_fin_raw = notice.get("deadline-receipt-request")
    fecha_fin_str = None
    if fecha_fin_raw:
        fecha_fin_str = str(fecha_fin_raw).strip()
        if "T" in fecha_fin_str:
            fecha_fin_str = fecha_fin_str.split("T")[0]
        if len(fecha_fin_str) >= 10:
            fecha_fin_str = fecha_fin_str[:10]

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
        tiene_ted = any(f.casefold() == "ted" for f in fuentes)
        if not tiene_ted:
            fuentes.append("TED")
            cambios["fuente"] = ", ".join(dict.fromkeys(fuentes))

        cambios_relevantes = "titulo" in cambios or "importe" in cambios or "fecha_fin" in cambios
        if cambios_relevantes:
            cambios["es_actualizada"] = True
            try:
                supabase.table("licitaciones").update(cambios).eq("id", reg_existente["id"]).execute()
                actualizadas += 1
            except Exception as e:
                print(f"Error actualizando {reg_existente['id']}: {e}")
        elif cambios:
            try:
                supabase.table("licitaciones").update(cambios).eq("id", reg_existente["id"]).execute()
            except Exception as e:
                print(f"Error actualizando metadatos {reg_existente['id']}: {e}")
        else:
            sin_cambios += 1
        continue

    clave = (titulo_limpio.casefold(), limpiar_organo(organo).casefold())
    registro_mismo_titulo = mapa_claves.get(clave)
    if registro_mismo_titulo is not None:
        fuente_existente = str(registro_mismo_titulo.get("fuente", "")).strip()
        fuentes = [f.strip() for f in fuente_existente.split(",") if f.strip()]
        tiene_ted = any(f.casefold() == "ted" for f in fuentes)
        if not tiene_ted:
            fuentes.append("TED")
            try:
                supabase.table("licitaciones").update({
                    "fuente": ", ".join(dict.fromkeys(fuentes))
                }).eq("id", registro_mismo_titulo["id"]).execute()
            except Exception as e:
                print(f"Error añadiendo TED como fuente: {e}")
        duplicados += 1
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
    except Exception as e:
        print(f"Error generando embedding para {publication_number}: {e}")
        elemento["embedding"] = None
    licitaciones_nuevas.append(elemento)
    nuevas += 1

# ============================================================
# ESTADÍSTICAS
# ============================================================
print()
print("=" * 60)
print("ESTADÍSTICAS TED")
print("=" * 60)
print(f"Duplicados evitados (ya estaban en PLACSP): {duplicados}")
print(f"Adjudicados/resultados filtrados: {resultados_filtrados}")
print(f"VEAT: {veat}")
print(f"Caducados: {caducados}")
print(f"Nuevas: {nuevas}")
print(f"Actualizadas: {actualizadas}")
print(f"Sin cambios: {sin_cambios}")

# ============================================================
# INSERTAR SOLO LAS NUEVAS
# ============================================================
if licitaciones_nuevas:
    print(f"Insertando {len(licitaciones_nuevas)} licitaciones nuevas...")
    BATCH_INSERT = 5
    for inicio in range(0, len(licitaciones_nuevas), BATCH_INSERT):
        lote = licitaciones_nuevas[inicio:inicio + BATCH_INSERT]
        try:
            supabase.table("licitaciones").insert(lote).execute()
            print(f"  -> Insertadas {min(inicio + BATCH_INSERT, len(licitaciones_nuevas))}/{len(licitaciones_nuevas)}")
        except Exception as e:
            print(f"Error insertando lote {inicio // BATCH_INSERT + 1}: {e}")
else:
    print("No hay licitaciones nuevas para insertar.")

# ============================================================
# LIMPIEZA DE TED CADUCADAS EN BBDD
# ============================================================
print()
print("Comprobando licitaciones TED caducadas en BBDD...")
try:
    ted_resp = supabase.table("licitaciones").select(
        "id,enlace,fuente,fecha_fin"
    ).ilike("fuente", "%TED%").execute()
    ted_bbdd = ted_resp.data or []
    eliminadas = 0
    fuentes_modificadas = 0

    for registro in ted_bbdd:
        fecha_fin = registro.get("fecha_fin")
        if not fecha_fin:
            continue
        try:
            fecha_fin_date = date.fromisoformat(str(fecha_fin)[:10])
        except ValueError:
            continue
        if fecha_fin_date >= HOY:
            continue

        fuente = str(registro.get("fuente", "")).strip()
        fuentes = [f.strip() for f in fuente.split(",") if f.strip()]
        fuentes_sin_ted = [f for f in fuentes if f.casefold() != "ted"]

        if not fuentes_sin_ted:
            try:
                supabase.table("licitaciones").delete().eq("id", registro["id"]).execute()
                eliminadas += 1
            except Exception as e:
                print(f"Error eliminando {registro['id']}: {e}")
        else:
            nueva_fuente = ", ".join(fuentes_sin_ted)
            try:
                supabase.table("licitaciones").update({
                    "fuente": nueva_fuente
                }).eq("id", registro["id"]).execute()
                fuentes_modificadas += 1
            except Exception as e:
                print(f"Error modificando fuente {registro['id']}: {e}")

    if eliminadas == 0 and fuentes_modificadas == 0:
        print("No hay licitaciones TED caducadas para eliminar.")
    else:
        print(f"Eliminadas por caducidad: {eliminadas}")
        print(f"Registros donde se eliminó TED de la fuente: {fuentes_modificadas}")
except Exception as e:
    print(f"Error comprobando caducadas en BBDD: {e}")

print()
print("=" * 60)
print("SINCRONIZACIÓN TED FINALIZADA")
print("=" * 60)
