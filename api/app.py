import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'cbc-place-ext': 'urn:dgpe:names:draft:codice:schema:xsd:ContractFolderStatusMessage-2-ext',
    'cac-place-ext': 'urn:dgpe:names:draft:codice:schema:xsd:ContractFolderStatusMessage-2-ext',
    'cbc': 'urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2',
    'cac': 'urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2'
}

FUENTES = [
    {
        "nombre": "Licitaciones Generales",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3.atom",
    },
    {
        "nombre": "Licitaciones Agregadas",
        "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_644/licitacionesAgregadas.atom",
    }
]

def _texto(el, xpath):
    nodo = el.find(xpath, NS)
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
            presupuesto_el = entry.find(".//cbc:PayableAmount", NS)

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

    descripcion = _texto(entry, ".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cbc:Name")
    if not descripcion:
        descripcion = _texto(entry, ".//cac:ProcurementProject/cbc:Description")
    if not descripcion:
        descripcion = titulo

    return {
        "titulo": titulo,
        "organo": organo,
        "fecha": fecha_str,
        "importe": importe,
        "enlace": enlace,
        "descripcion": descripcion
    }

@app.route('/api/buscar', methods=['GET'])
def api_buscar():
    query = request.args.get('q', '').lower().strip()
    licitaciones = []
    headers = {'User-Agent': 'Mozilla/5.0'}

    for fuente in FUENTES:
        try:
            resp = requests.get(fuente["url"], headers=headers, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for entry in root.findall("atom:entry", NS):
                    datos = parsear_entrada_xml(entry)
                    licitaciones.append(datos)
        except Exception as e:
            print(f"Error leyendo fuente: {e}")

    if not licitaciones:
        return jsonify([])

    df = pd.DataFrame(licitaciones)
    df = df.drop_duplicates(subset=["enlace"]).reset_index(drop=True)

    if query:
        mask = (
            df["titulo"].str.lower().str.contains(query, na=False) | 
            df["descripcion"].str.lower().str.contains(query, na=False) |
            df["organo"].str.lower().str.contains(query, na=False)
        )
        df = df[mask]

    if df.empty:
        return jsonify([])

    df = df.sort_values(by="fecha", ascending=False).head(100).reset_index(drop=True)
    
    resultados = []
    for idx, row in df.iterrows():
        imp_fmt = f"{row['importe']:,.2f} €" if row['importe'] > 0 else "No especificado"
        resultados.append({
            "id": idx + 1,
            "titulo": row["titulo"],
            "organo": row["organo"] if row["organo"] else "No especificado",
            "fecha": row["fecha"],
            "importe": imp_fmt,
            "enlace": row["enlace"]
        })

    return jsonify(resultados)

@app.route('/')
def index():
    html_template = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Buscador Inteligente de Licitaciones PLACSP</title>
        <style>
            :root { --primary: #2563eb; --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #64748b; --border: #e2e8f0; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; background: var(--card); padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h1 { font-size: 1.8rem; margin-bottom: 5px; }
            p.subtitle { color: var(--muted); margin-bottom: 25px; }
            .search-box { display: flex; gap: 12px; margin-bottom: 20px; }
            input[type="text"] { flex: 1; padding: 14px 18px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 1rem; outline: none; }
            input[type="text"]:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
            button { background: var(--primary); color: white; border: none; padding: 0 28px; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; }
            button:hover { background: #1d4ed8; }
            #loading { display: none; text-align: center; color: var(--muted); font-style: italic; margin: 25px 0; }
            .table-responsive { overflow-x: auto; margin-top: 15px; border-radius: 8px; border: 1px solid var(--border); }
            table { width: 100%; border-collapse: collapse; font-size: 0.92rem; text-align: left; }
            th, td { padding: 14px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }
            th { background: #f1f5f9; color: #334155; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }
            tr:hover { background: #f8fafc; }
            .badge { background: #e0f2fe; color: #0369a1; padding: 6px 10px; border-radius: 6px; font-weight: 600; white-space: nowrap; }
            a.link { color: var(--primary); text-decoration: none; font-weight: 600; white-space: nowrap; }
            a.link:hover { text-decoration: underline; }
            #contador { font-size: 0.9rem; color: var(--muted); margin-bottom: 15px; font-weight: 500; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏛️ Buscador Inteligente de Licitaciones (PLACSP)</h1>
            <p class="subtitle">Desplegado en Vercel con motor Python en tiempo real.</p>
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Ej. software, limpieza, Madrid, mantenimiento..." onkeypress="if(event.key==='Enter') buscar()">
                <button onclick="buscar()">🔍 Buscar</button>
            </div>
            <div id="loading">⏳ Consultando feeds oficiales en servidor... por favor espera...</div>
            <div id="contador"></div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 4%;">#</th>
                            <th style="width: 36%;">Título / Objeto</th>
                            <th style="width: 25%;">Órgano Contratante</th>
                            <th style="width: 10%;">Fecha</th>
                            <th style="width: 12%;">Importe</th>
                            <th style="width: 13%;">Enlace</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        <tr><td colspan="6" style="text-align: center; color: var(--muted); padding: 30px;">Escribe un término y pulsa buscar.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        <script>
            async function buscar() {
                const q = document.getElementById('searchInput').value.trim();
                const loading = document.getElementById('loading');
                const tbody = document.getElementById('tableBody');
                const contador = document.getElementById('contador');

                loading.style.display = 'block';
                tbody.innerHTML = '';
                contador.innerHTML = '';

                try {
                    const res = await fetch(`/api/buscar?q=${encodeURIComponent(q)}`);
                    const data = await res.json();
                    loading.style.display = 'none';

                    if (data.length === 0) {
                        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--muted); padding: 30px;">No se encontraron licitaciones.</td></tr>`;
                        contador.innerHTML = "Se encontraron 0 resultados.";
                        return;
                    }

                    contador.innerHTML = `Mostrando ${data.length} licitaciones encontradas.`;
                    let html = "";
                    data.forEach(item => {
                        html += `
                            <tr>
                                <td><strong>${item.id}</strong></td>
                                <td>${item.titulo}</td>
                                <td>${item.organo}</td>
                                <td>${item.fecha}</td>
                                <td><span class="badge">${item.importe}</span></td>
                                <td><a href="${item.enlace}" target="_blank" class="link">Ver en PLACSP 🔗</a></td>
                            </tr>
                        `;
                    });
                    tbody.innerHTML = html;
                } catch (err) {
                    loading.style.display = 'none';
                    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: red; padding: 30px;">Error al conectar con el servidor.</td></tr>`;
                }
            }
            window.onload = () => buscar();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True)