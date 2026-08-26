# -*- coding: utf-8 -*-
"""
Buscador de Licitaciones PLACSP - Servidor FastAPI (Versión Ligera para Vercel)
"""

import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

app = FastAPI(title="Buscador de Licitaciones PLACSP", version="2.1")

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
    {"nombre": "Licitaciones Generales", "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3.atom"},
    {"nombre": "Licitaciones Agregadas", "url": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_644/licitacionesAgregadas.atom"}
]

GLOBAL_DF = pd.DataFrame()

def cargar_cache():
    global GLOBAL_DF
    try:
        if os.path.exists(CACHE_DB):
            GLOBAL_DF = pd.read_pickle(CACHE_DB)
    except Exception as e:
        print(f"Error cargando caché: {e}")

cargar_cache()

@app.get("/", response_class=HTMLResponse)
def read_root():
    total_licitaciones = len(GLOBAL_DF)
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
            <p class="text-gray-600 mt-2">Versión optimizada para la nube.</p>
            <div class="inline-block mt-3 px-4 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                Base de datos activa: {total_licitaciones} licitaciones
            </div>
        </header>
        <div class="bg-white p-6 rounded-xl shadow-md text-center">
            <p class="text-gray-700">El servidor FastAPI está funcionando correctamente en Vercel.</p>
        </div>
    </div>
</body>
</html>"""

@app.post("/api/buscar")
def api_buscar(payload: dict):
    global GLOBAL_DF
    if GLOBAL_DF.empty:
        return {"resultados": [], "mensaje": "La base de datos está vacía. Sincroniza novedades."}

    df = GLOBAL_DF.copy()
    consulta = payload.get("consulta", "").lower()
    
    if consulta:
        df = df[df["titulo"].str.lower().str.contains(consulta, na=False) | df["organo"].str.lower().str.contains(consulta, na=False)]

    resultados = []
    for _, row in df.head(50).iterrows():
        resultados.append({
            "titulo": row.get("titulo", ""),
            "organo": row.get("organo", ""),
            "fecha": row.get("fecha", ""),
            "importe": row.get("importe", 0.0),
            "enlace": row.get("enlace", ""),
            "relevancia": 100.0
        })

    return {"resultados": resultados, "mensaje": f"Se encontraron {len(resultados)} licitaciones."}
