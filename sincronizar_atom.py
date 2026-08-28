import os 
import requests
import lxml.etree as ET
from datetime import datetime
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# 1. Configuración de credenciales desde las Variables de Entorno de GitHub Actions
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Cargar modelo de IA en CPU
print("Cargando modelo de IA (multilingual-e5-small)...")
encoder = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')

# Namespaces para leer la estructura CODICE perfectamente
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cac": "urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2",
    "cac-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2",
    "cbc-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2",
}

# Diccionario de correspondencias de códigos NUTS 2 y 3 de España a nombres legibles
MAPEO_NUTS = {
    # --- NUTS 2 (Comunidades y Ciudades Autónomas) ---
    "ES11": "Galicia",
    "ES12": "Principado de Asturias",
    "ES13": "Cantabria",
    "ES21": "País Vasco",
    "ES22": "Comunidad Foral de Navarra",
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

    # --- NUTS 3 (Provincias e Islas) ---
    "ES111": "A Coruña", "ES112": "Lugo", "ES113": "Ourense", "ES114": "Pontevedra",
    "ES120": "Asturias", "ES130": "Cantabria",
    "ES211": "Álava", "ES212": "Guipúzcoa", "ES213": "Vizcaya",
    "ES220": "Navarra", "ES230": "La Rioja",
    "ES241": "Huesca", "ES242": "Teruel", "ES243": "Zaragoza",
    "ES300": "Madrid",
    "ES411": "Ávila", "ES412": "Burgos", "ES413": "León", "ES414": "Palencia",
    "ES415": "Salamanca", "ES416": "Segovia", "ES417": "Soria", "ES418": "Valladolid", "ES419": "Zamora",
    "ES421": "Albacete", "ES422": "Ciudad Real", "ES423": "Cuenca", "ES424": "Guadalajara", "ES425": "Toledo",
    "ES431": "Badajoz", "ES432": "Cáceres",
    "ES511": "Barcelona", "ES512": "Girona", "ES513": "Lleida", "ES514": "Tarragona",
    "ES521": "Alicante", "ES522": "Castellón", "ES523": "Valencia",
    "ES531": "Eivissa i Formentera", "ES532": "Mallorca", "ES533": "Menorca",
    "ES611": "Almería", "ES612": "Cádiz", "ES613": "Córdoba", "ES614": "Granada",
    "ES615": "Huelva", "ES616": "Jaén", "ES617": "Málaga", "ES618": "Sevilla",
    "ES620": "Murcia", "ES630": "Ceuta", "ES640": "Melilla",
    "ES703": "El Hierro", "ES704": "Fuerteventura", "ES705": "Gran Canaria",
    "ES706": "La Gomera", "ES707": "La Palma", "ES708": "Lanzarote", "ES709": "Tenerife"
}

FUENTES_ATOM = [
    {
        "nombre": "Licitaciones Generales",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3.atom"
    },
    {
        "nombre": "Licitaciones Agregadas",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_644/licitacionesAgregadas.atom"
    }
]

def _texto(el, xpath, ns=NS):
    nodo = el.find(xpath, ns)
    return nodo.text.strip() if nodo is not None and nodo.text else None

def sincronizar():
    print("🔄 Iniciando sincronización avanzada de feeds ATOM...")
    
    for fuente in FUENTES_ATOM:
        print(f"📥 Leyendo fuente: {fuente['nombre']}...")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            response = requests.get(fuente["url"], headers=headers, timeout=45)
            response.raise_for_status()
            
            parser = ET.XMLParser(recover=True)
            root = ET.fromstring(response.content, parser=parser)
        except Exception as e:
            print(f"❌ Error descargando {fuente['nombre']}: {e}")
            continue

        entries = root.findall('atom:entry', NS)
        if not entries:
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')

        print(f"Procesando {len(entries)} entradas de {fuente['nombre']}...")

        for entry in entries:
            titulo = _texto(entry, "atom:title") or "Sin título"
            
            # Extracción del enlace
            enlace_el = entry.find("atom:link", NS)
            enlace = enlace_el.get("href") if enlace_el is not None else ""
            if not enlace:
                enlace_el = entry.find('.//{http://www.w3.org/2005/Atom}link')
                enlace = enlace_el.get("href") if enlace_el is not None else ""
            
            if not enlace:
                continue

            # Fechas
            txt_fecha = _texto(entry, "atom:updated") or _texto(entry, "atom:published")
            fecha = txt_fecha.split("T")[0] if txt_fecha else str(datetime.now().date())[:10]

            # --- Fecha fin de presentación de oferta ---
            fecha_fin = "No especificada"
            try:
                end_date_el = entry.find(".//cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod/cbc:EndDate", NS)
                if end_date_el is not None and end_date_el.text:
                    fecha_fin = end_date_el.text.strip()
            except Exception:
                pass

            # --- Lugar de ejecución (con traducción automática de códigos NUTS) ---
            lugar_ejecucion = "No especificado"
            try:
                lugar_el = entry.find(".//cac:ProcurementProject/cac:RealizedLocation/cbc:CountrySubentity", NS)
                if lugar_el is not None and lugar_el.text:
                    lugar_ejecucion = lugar_el.text.strip()
                else:
                    lugar_el = entry.find(".//cac:ProcurementProject/cac:RealizedLocation/cbc:CountrySubentityCode", NS)
                    if lugar_el is not None and lugar_el.text:
                        codigo_nuts = lugar_el.text.strip()
                        lugar_ejecucion = MAPEO_NUTS.get(codigo_nuts, codigo_nuts)
            except Exception:
                pass

            # --- Importe (Búsqueda robusta en cascada) ---
            importe = 0.0
            try:
                presupuesto_el = entry.find(".//cac:BudgetAmount/cbc:EstimatedOverallContractAmount", NS)
                if presupuesto_el is None:
                    presupuesto_el = entry.find(".//cac:BudgetAmount/cbc:TaxExclusiveAmount", NS)
                if presupuesto_el is None:
                    presupuesto_el = entry.find(".//cac:BudgetAmount/cbc:TotalAmount", NS)
                
                if presupuesto_el is not None and presupuesto_el.text:
                    importe = float(presupuesto_el.text.strip().replace(",", "."))
                else:
                    lotes = entry.findall(".//cac:ProcurementProjectLot", NS)
                    if lotes:
                        suma_lotes = 0.0
                        for lote in lotes:
                            lote_amt = lote.find(".//cac:BudgetAmount/cbc:TaxExclusiveAmount", NS)
                            if lote_amt is None:
                                lote_amt = lote.find(".//cac:BudgetAmount/cbc:TotalAmount", NS)
                            if lote_amt is not None and lote_amt.text:
                                suma_lotes += float(lote_amt.text.strip().replace(",", "."))
                        if suma_lotes > 0:
                            importe = suma_lotes
            except Exception:
                importe = 0.0

            # --- Órgano contratante ---
            organo = "Órgano desconocido"
            try:
                organo_el = entry.find(".//cac-place-ext:LocatedContractingParty//cac:PartyName//cbc:Name", NS)
                if organo_el is None:
                    organo_el = entry.find(".//cac:ContractingParty//cbc:Name", NS)
                if organo_el is not None and organo_el.text:
                    organo = organo_el.text.strip()
            except Exception:
                pass

            # Descripción adicional
            descripcion = _texto(entry, ".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cbc:Name", NS)
            if not descripcion:
                descripcion = _texto(entry, ".//cac:ProcurementProject/cbc:Description", NS) or ""

            # Texto enriquecido para la IA
            texto_evaluacion = (
                f"passage: Título: {titulo}. "
                f"Objeto del contrato: {descripcion}. "
                f"Órgano: {organo}. "
                f"Lugar de ejecución: {lugar_ejecucion}. "
                f"Importe: {importe} EUR. "
                f"Fecha fin oferta: {fecha_fin}"
            )

            # Verificar si ya existe en Supabase
            resp = supabase.table("licitaciones").select("id, enlace").eq("enlace", enlace).execute()

            if not resp.data:
                # --- CASO 1: NUEVA LICITACIÓN ---
                vector = encoder.encode(texto_evaluacion).tolist()
                nuevo_registro = {
                    "titulo": titulo.strip(),
                    "organo": organo.strip(),
                    "fecha": fecha,
                    "importe": importe,
                    "enlace": enlace.strip(),
                    "lugar_ejecucion": lugar_ejecucion,
                    "fecha_fin": fecha_fin,
                    "texto_completo": texto_evaluacion,
                    "embedding": vector
                }
                supabase.table("licitaciones").insert(nuevo_registro).execute()
                print(f"➕ Nueva añadida: {titulo[:30]}... ({importe}€)")
            else:
                # --- CASO 2: ACTUALIZACIÓN DE LICITACIÓN YA EXISTENTE ---
                vector = encoder.encode(texto_evaluacion).tolist()
                datos_actualizados = {
                    "titulo": titulo.strip(),
                    "organo": organo.strip(),
                    "fecha": fecha,
                    "importe": importe,
                    "lugar_ejecucion": lugar_ejecucion,
                    "fecha_fin": fecha_fin,
                    "texto_completo": texto_evaluacion,
                    "embedding": vector
                }
                supabase.table("licitaciones").update(datos_actualizados).eq("enlace", enlace).execute()
                print(f"🔄 Actualizada: {titulo[:30]}... ({importe}€)")

    print("✅ Sincronización ATOM avanzada completada con éxito.")

if __name__ == "__main__":
    sincronizar()
