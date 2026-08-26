from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd

app = FastAPI(title="Buscador de Licitaciones PLACSP", version="2.4")

DATA_DIR = "/tmp/licitaciones_data"
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_DB = os.path.join(DATA_DIR, "licitaciones_db.pkl")

NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'cbc-place-ext': 'urn:dgpe:names:draft:codice:schema:xsd:ContractFolderStatusMessage-2-ext',
    'cac-place-ext': 'urn:dgpe:names:draft:codice:schema:xsd:ContractFolderStatusMessage-2-ext',
    'cbc': 'urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2',
    'cac': 'urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2'
}

FUENTES = [
    {"nombre": "Generales", "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3.atom"},
    {"nombre": "Agregadas", "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_644/licitacionesAgregadas.atom"}
]

def obtener_df():
    try:
        if os.path.exists(CACHE_DB):
            df = pd.read_pickle(CACHE_DB)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
    except Exception:
        pass
    return pd.DataFrame(columns=["titulo", "organo", "fecha", "importe", "enlace"])

def guardar_df(df):
    try:
        df.to_pickle(CACHE_DB)
    except Exception:
        pass

def _texto(el, xpath, ns=NS):
    nodo = el.find(xpath, ns)
    return nodo.text.strip() if nodo is not None and nodo.text else ""

def parsear_entrada_xml(entry):
    titulo = _texto(entry, "atom:title")
    enlace_el = entry.find("atom:link", NS)
    enlace = enlace_el.get("href") if enlace_el is not None else ""
    txt_fecha = _texto(entry, "atom:updated") or _texto(entry, "atom:published")
    fecha_str = txt_fecha.split("T")[0] if txt_fecha else ""

    importe = 0.0
    try:
        presupuesto_el = entry.find(".//cbc-place-ext:EstimatedOverallContractAmount", NS) or \
                         entry.find(".//cbc:EstimatedOverallContractAmount", NS) or \
                         entry.find(".//cbc:TaxExclusiveAmount", NS) or \
                         entry.find(".//cbc:TotalAmount", NS)
        if presupuesto_el is not None and presupuesto_el.text:
            importe = float(presupuesto_el.text.strip().replace(".", "").replace(",", "."))
    except Exception:
        pass

    organo = ""
    try:
        organo_el = entry.find(".//cac-place-ext:ContractingParty//cbc-place-ext:PartyName//cbc-place-ext:Name", NS) or \
                    entry.find(".//cac:ContractingParty//cbc:Name", NS)
        if organo_el is not None and organo_el.text:
            organo = organo_el.text.strip()
    except Exception:
        pass

    return {"titulo": titulo, "organo": organo, "fecha": fecha_str, "importe": importe, "enlace": enlace}

@app.get("/", response_class=HTMLResponse)
def read_root():
    df = obtener_df()
    total_licitaciones = len(df)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Buscador de Licitaciones PLACSP</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 font-sans antialiased">
    <div class="max-w-7xl mx-auto px-4 py-8">
        <header class="mb-8 text-center">
            <h1 class="text-3xl font-extrabold text-blue-900">Buscador de Licitaciones Públicas (PLACSP)</h1>
            <p class="text-gray-600 mt-2">Interfaz web optimizada para Vercel.</p>
            <div class="inline-block mt-3 px-4 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                Base de datos activa: {total_licitaciones} licitaciones
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-4 gap-8">
            <div class="bg-white p-6 rounded-xl shadow-md lg:col-span-1 space-y-4">
                <h2 class="text-lg font-bold text-gray-800 border-b pb-2">Filtros</h2>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Texto en Título</label>
                    <input type="text" id="consulta" placeholder="ej. obras, limpieza..." class="mt-1 w-full p-2 border border-gray-300 rounded-lg">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Órgano Contratante</label>
                    <input type="text" id="organo" placeholder="ej. Ayuntamiento..." class="mt-1 w-full p-2 border border-gray-300 rounded-lg">
                </div>
                <button onclick="realizarBusqueda()" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg shadow">
                    🔍 Buscar
                </button>
                <button onclick="buscarNovedades()" class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded-lg shadow">
                    🌐 Sincronizar Novedades
                </button>
                <button onclick="exportarExcel()" class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg shadow">
                    📥 Descargar Excel
                </button>
            </div>

            <div class="bg-white p-6 rounded-xl shadow-md lg:col-span-3">
                <div class="flex justify-between items-center mb-4 border-b pb-2">
                    <h2 class="text-lg font-bold text-gray-800">Resultados</h2>
                    <span id="resultado-info" class="text-sm text-gray-600">Pulsa buscar o sincroniza.</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">#</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Título</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Órgano</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Fecha</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Importe</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Enlace</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200" id="resultados-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function realizarBusqueda() {
            const consulta = document.getElementById('consulta').value;
            const organo = document.getElementById('organo').value;
            const response = await fetch('/api/buscar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.xml ? null : JSON.stringify({ consulta, organo })
            });
            const data = await response.json();
            document.getElementById('resultado-info').innerText = data.mensaje;
            const tbody = document.getElementById('resultados-body');
            tbody.innerHTML = '';
            data.resultados.forEach((row, index) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="px-4 py-3 text-sm text-gray-500">${index + 1}</td>
                    <td class="px-4 py-3 text-sm text-gray-900 font-medium">${row.titulo}</td>
                    <td class="px-4 py-3 text-sm text-gray-600">${row.organo}</td>
                    <td class="px-4 py-3 text-sm text-gray-600">${row.fecha}</td>
                    <td class="px-4 py-3 text-sm text-gray-600">${row.importe.toLocaleString()} €</td>
                    <td class="px-4 py-3 text-sm"><a href="${row.enlace}" target="_blank" class="text-blue-600 hover:underline font-bold">Ver 🔗</a></td>
                `;
                tbody.appendChild(tr);
            });
        }
        async function buscarNovedades() {
            alert("Conectando con PLACSP...");
            const response = await fetch('/api/actualizar', { method: 'POST' });
            const data = await response.json();
            alert(data.mensaje);
            location.reload();
        }
        async function exportarExcel() {
            window.location.href = '/api/exportar';
        }
    </script>
</body>
</html>"""

@app.post("/api/buscar")
def api_buscar(payload: dict):
    df = obtener_df()
    if df.empty:
        return {"resultados": [], "mensaje": "La base de datos está vacía. Sincroniza novedades."}
    
    consulta = payload.get("consulta", "").lower()
    organo = payload.get("organo", "").lower()
    
    if consulta:
        df = df[df["titulo"].str.lower().str.contains(consulta, na=False)]
    if organo:
        df = df[df["organo"].str.lower().str.contains(organo, na=False)]
        
    resultados = df.head(100).to_dict(orient="records")
    return {"resultados": resultados, "mensaje": f"Se encontraron {len(resultados)} licitaciones."}

@app.post("/api/actualizar")
def api_actualizar():
    df_actual = obtener_df()
    enlaces_conocidos = set(df_actual["enlace"]) if not df_actual.empty else set()
    nuevas = []

    for fuente in FUENTES:
        try:
            resp = requests.get(fuente["url"], headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for entry in root.findall("atom:entry", NS):
                    datos = parsear_entrada_xml(entry)
                    if datos["enlace"] and datos["enlace"] not in enlaces_conocidos:
                        nuevas.append(datos)
                        enlaces_conocidos.add(datos["enlace"])
        except Exception:
            pass

    if not nuevas:
        return {"mensaje": "La base de datos ya está al día."}

    df_nuevas = pd.DataFrame(nuevas)
    df_final = pd.concat([df_nuevas, df_actual], ignore_index=True) if not df_actual.empty else df_nuevas
    guardar_df(df_final)

    return {"mensaje": f"¡Actualización exitosa! +{len(nuevas)} licitaciones añadidas."}

@app.get("/api/exportar")
def api_exportar():
    df = obtener_df()
    if df.empty:
        return {"error": "No hay datos"}
    ruta_excel = os.path.join(DATA_DIR, "licitaciones.xlsx")
    df.to_excel(ruta_excel, index=False, engine='openpyxl')
    return FileResponse(ruta_excel, filename="licitaciones.xlsx")
