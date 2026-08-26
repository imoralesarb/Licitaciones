# -*- coding: utf-8 -*-
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from fastapi.responses import HTMLResponse, FileResponse

NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'cbc-place-ext': 'urn:dgpe:names:draft:codice:schema:xsd:ContractFolderStatusMessage-2-ext',
    'cac-place-ext': 'urn:dgpe:names:draft:codice:schema:xsd:ContractFolderStatusMessage-2-ext',
    'cbc': 'urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2',
    'cac': 'urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2'
}

FUENTES = [
    {
        "nombre": "1. Licitaciones Generales (Perfiles de Contratante)",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3.atom",
    },
    {
        "nombre": "2. Licitaciones Agregadas (CCAA y EELL)",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_644/licitacionesAgregadas.atom",
    },
]

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
        presupuesto_el = entry.find(".//cbc-place-ext:EstimatedOverallContractAmount", NS)
        if presupuesto_el is None:
            presupuesto_el = entry.find(".//cbc:EstimatedOverallContractAmount", NS)
        if presupuesto_el is None:
            presupuesto_el = entry.find(".//cbc:TaxExclusiveAmount", NS)
        if presupuesto_el is None:
            presupuesto_el = entry.find(".//cbc:TotalAmount", NS)
        
        if presupuesto_el is not None and presupuesto_el.text:
            texto_importe = presupuesto_el.text.strip().replace(".", "").replace(",", ".")
            importe = float(texto_importe)
    except Exception:
        pass

    organo = ""
    try:
        organo_el = entry.find(".//cac-place-ext:ContractingParty//cbc-place-ext:PartyName//cbc-place-ext:Name", NS)
        if organo_el is None:
            organo_el = entry.find(".//cac:ContractingParty//cbc:Name", NS)
        organo = organo_el.text.strip() if organo_el is not None and organo_el.text else ""
    except Exception:
        pass

    return {
        "titulo": titulo,
        "organo": organo,
        "fecha": fecha_str,
        "importe": importe,
        "enlace": enlace
    }

def registrar_rutas(app, global_df_ref, data_dir, cache_db):

    @app.get("/", response_class=HTMLResponse)
    def read_root():
        total_licitaciones = len(global_df_ref)
        html_content = f"""<!DOCTYPE html>
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
            <p class="text-gray-600 mt-2">Filtros por texto, importe, fechas y órgano.</p>
            <div id="status-badge" class="inline-block mt-3 px-4 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                Base de datos activa: {total_licitaciones} licitaciones
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-4 gap-8">
            <!-- PANEL DE FILTROS -->
            <div class="bg-white p-6 rounded-xl shadow-md lg:col-span-1 space-y-4">
                <h2 class="text-lg font-bold text-gray-800 border-b pb-2">Filtros de Búsqueda</h2>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Texto en Título</label>
                    <input type="text" id="consulta" placeholder="ej. mantenimiento..." class="mt-1 w-full p-2 border border-gray-300 rounded-lg">
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-700">Órgano Contratante</label>
                    <input type="text" id="organo" placeholder="ej. Ayuntamiento..." class="mt-1 w-full p-2 border border-gray-300 rounded-lg">
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <label class="block text-xs font-medium text-gray-700">Fecha Desde</label>
                        <input type="date" id="fecha_desde" class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-gray-700">Fecha Hasta</label>
                        <input type="date" id="fecha_hasta" class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm">
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <label class="block text-xs font-medium text-gray-700">Importe Mín (€)</label>
                        <input type="number" id="importe_min" value="0" class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-gray-700">Importe Máx (€)</label>
                        <input type="number" id="importe_max" value="0" class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm">
                    </div>
                </div>

                <button onclick="realizarBusqueda()" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg shadow">
                    🔍 Buscar Licitaciones
                </button>

                <button onclick="buscarNovedades()" class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded-lg shadow">
                    🌐 Sincronizar Novedades
                </button>

                <button onclick="exportarExcel()" class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg shadow">
                    📥 Descargar Excel
                </button>
            </div>

            <!-- RESULTADOS -->
            <div class="bg-white p-6 rounded-xl shadow-md lg:col-span-3">
                <div class="flex justify-between items-center mb-4 border-b pb-2">
                    <h2 class="text-lg font-bold text-gray-800">Resultados</h2>
                    <span id="resultado-info" class="text-sm text-gray-600">Introduce una consulta o pulsa buscar.</span>
                </div>

                <div id="loading" class="hidden text-center py-12">
                    <div class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
                    <p class="text-gray-600 mt-2">Buscando...</p>
                </div>

                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">#</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Título y Objeto</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Órgano</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Fecha</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Importe</th>
                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Enlace</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200" id="resultados-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function realizarBusqueda() {
            document.getElementById('loading').classList.remove('hidden');
            const consulta = document.getElementById('consulta').value;
            const organo = document.getElementById('organo').value;
            const fecha_desde = document.getElementById('fecha_desde').value;
            const fecha_hasta = document.getElementById('fecha_hasta').value;
            const importe_min = document.getElementById('importe_min').value;
            const importe_max = document.getElementById('importe_max').value;

            const response = await fetch('/api/buscar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ consulta, organo, fecha_desde, fecha_hasta, importe_min: parseFloat(importe_min), importe_max: parseFloat(importe_max) })
            });

            const data = await response.json();
            document.getElementById('loading').classList.add('hidden');
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
            alert("Conectando con PLACSP para descargar novedades...");
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
        return html_content

    @app.post("/api/buscar")
    def api_buscar(payload: dict):
        if global_df_ref.empty:
            return {"resultados": [], "mensaje": "La base de datos está vacía. Pulsa 'Sincronizar Novedades'."}

        df = global_df_ref.copy()
        total_leidas = len(df)

        consulta = payload.get("consulta", "").lower()
        organo_filtro = payload.get("organo", "").lower()
        f_desde = payload.get("fecha_desde", "")
        f_hasta = payload.get("fecha_hasta", "")
        i_min = payload.get("importe_min", 0)
        i_max = payload.get("importe_max", 0)

        if f_desde:
            df = df[df["fecha"] >= f_desde]
        if f_hasta:
            df = df[df["fecha"] <= f_hasta]
        if i_min and i_min > 0:
            df = df[df["importe"] >= i_min]
        if i_max and i_max > 0:
            df = df[df["importe"] <= i_max]
        if organo_filtro:
            df = df[df["organo"].str.lower().str.contains(organo_filtro, na=False)]
        if consulta:
            df = df[df["titulo"].str.lower().str.contains(consulta, na=False)]

        if df.empty:
            return {"resultados": [], "mensaje": f"No hay resultados con esos filtros sobre {total_leidas} licitaciones."}

        resultados = []
        for _, row in df.head(100).iterrows():
            resultados.append({
                "titulo": row.get("titulo", ""),
                "organo": row.get("organo", ""),
                "fecha": row.get("fecha", ""),
                "importe": row.get("importe", 0.0),
                "enlace": row.get("enlace", "")
            })

        return {
            "resultados": resultados,
            "mensaje": f"Se encontraron {len(resultados)} licitaciones (analizadas {total_leidas})."
        }

    @app.post("/api/actualizar")
    def api_actualizar():
        nonlocal global_df_ref
        enlaces_conocidos = set(global_df_ref["enlace"]) if not global_df_ref.empty else set()
        nuevas_totales = []

        headers = {'User-Agent': 'Mozilla/5.0'}
        for fuente in FUENTES:
            try:
                resp = requests.get(fuente["url"], headers=headers, timeout=30)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for entry in root.findall("atom:entry", NS):
                        datos = parsear_entrada_xml(entry)
                        if datos["enlace"] and datos["enlace"] not in enlaces_conocidos:
                            nuevas_totales.append(datos)
                            enlaces_conocidos.add(datos["enlace"])
            except Exception as e:
                print(f"Error en fuente: {e}")

        if not nuevas_totales:
            return {"mensaje": "La base de datos ya está al día. No hay novedades."}

        df_nuevas = pd.DataFrame(nuevas_totales)

        if global_df_ref.empty:
            global_df_ref = df_nuevas
        else:
            global_df_ref = pd.concat([df_nuevas, global_df_ref], ignore_index=True)

        try:
            global_df_ref.to_pickle(cache_db)
        except Exception:
            pass

        return {"mensaje": f"¡Actualización exitosa! Se agregaron +{len(df_nuevas)} licitaciones nuevas."}

    @app.get("/api/exportar")
    def api_exportar():
        if global_df_ref.empty:
            return {"error": "No hay datos para exportar"}
        ruta_excel = os.path.join(data_dir, "licitaciones_exportadas.xlsx")
        global_df_ref.to_excel(ruta_excel, index=False, engine='openpyxl')
        return FileResponse(ruta_excel, filename="licitaciones.xlsx")
