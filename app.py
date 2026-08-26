import os
import time
import zipfile
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import requests
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

app = FastAPI(title="Buscador de Licitaciones PLACSP", version="2.0")

DATA_DIR = "/tmp/licitaciones_data"
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_DB = os.path.join(DATA_DIR, "licitaciones_db.pkl")
CACHE_EMB = os.path.join(DATA_DIR, "licitaciones_embeddings.npy")

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

GLOBAL_DF = pd.DataFrame()
GLOBAL_EMBEDDINGS = None
encoder = None


def get_encoder():
    global encoder

    if encoder is None:
        print("Cargando modelo semántico (multilingual-e5-small)...")
        encoder = SentenceTransformer(
            'intfloat/multilingual-e5-small',
            device='cpu'
        )

    return encoder


def _texto(el, xpath, ns=NS):
    nodo = el.find(xpath, ns)

    return nodo.text.strip() if nodo is not None and nodo.text else ""


def parsear_entrada_xml(entry):
    titulo = _texto(entry, "atom:title")

    enlace_el = entry.find("atom:link", NS)
    enlace = enlace_el.get("href") if enlace_el is not None else ""

    txt_fecha = (
        _texto(entry, "atom:updated")
        or _texto(entry, "atom:published")
    )

    fecha_str = txt_fecha.split("T")[0] if txt_fecha else ""

    importe = 0.0

    try:
        presupuesto_el = entry.find(
            ".//cbc-place-ext:EstimatedOverallContractAmount",
            NS
        )

        if presupuesto_el is None:
            presupuesto_el = entry.find(
                ".//cbc:EstimatedOverallContractAmount",
                NS
            )

        if presupuesto_el is None:
            presupuesto_el = entry.find(
                ".//cbc:TaxExclusiveAmount",
                NS
            )

        if presupuesto_el is None:
            presupuesto_el = entry.find(
                ".//cbc:TotalAmount",
                NS
            )

        if presupuesto_el is None:
            presupuesto_el = entry.find(
                ".//cbc:PayableAmount",
                NS
            )

        if presupuesto_el is None:
            presupuesto_el = entry.find(
                ".//cac:BudgetAmount/cbc:TaxExclusiveAmount",
                NS
            )

        if presupuesto_el is not None and presupuesto_el.text:
            texto_importe = presupuesto_el.text.strip()

            if "," in texto_importe and "." in texto_importe:
                texto_importe = texto_importe.replace(".", "")
                texto_importe = texto_importe.replace(",", ".")

            elif "," in texto_importe:
                texto_importe = texto_importe.replace(",", ".")

            importe = float(texto_importe)

    except Exception:
        pass

    organo = ""

    try:
        organo_el = entry.find(
            ".//cac-place-ext:ContractingParty//cbc-place-ext:PartyName//cbc-place-ext:Name",
            NS
        )

        if organo_el is None:
            organo_el = entry.find(
                ".//cac:ContractingParty//cbc:Name",
                NS
            )

        organo = (
            organo_el.text.strip()
            if organo_el is not None and organo_el.text
            else ""
        )

    except Exception:
        pass

    descripcion = _texto(
        entry,
        ".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cbc:Name",
        NS
    )

    if not descripcion:
        descripcion = _texto(
            entry,
            ".//cac:ProcurementProject/cbc:Description",
            NS
        )

    lugar_ejecucion = ""

    try:
        realized_loc = entry.find(
            ".//cac-place-ext:ContractFolderStatus/cac:ProcurementProject/cac:RealizedLocation",
            NS
        )

        if realized_loc is not None:
            nuts = _texto(
                realized_loc,
                ".//cbc:CountrySubentityCode",
                NS
            )

            ciudad = _texto(
                realized_loc,
                ".//cbc:CityName",
                NS
            )

            pais = _texto(
                realized_loc,
                ".//cac:Country/cbc:IdentificationCode",
                NS
            )

            partes_lugar = [
                p for p in [ciudad, nuts, pais]
                if p
            ]

            lugar_ejecucion = (
                " - ".join(partes_lugar)
                if partes_lugar
                else ""
            )

    except Exception:
        pass

    plazo_ofertas = ""

    try:
        deadline_el = entry.find(
            ".//cac-place-ext:ContractFolderStatus/cac:TenderingProcess/cac:TenderSubmissionDeadlinePeriod",
            NS
        )

        if deadline_el is not None:
            f_fin = _texto(
                deadline_el,
                "cbc:EndDate",
                NS
            )

            h_fin = _texto(
                deadline_el,
                "cbc:EndTime",
                NS
            )

            desc_plazo = _texto(
                deadline_el,
                "cbc:Description",
                NS
            )

            if f_fin:
                plazo_ofertas = f"{f_fin} {h_fin}".strip()

            elif desc_plazo:
                plazo_ofertas = desc_plazo

    except Exception:
        pass

    texto_evaluacion = (
        f"passage: Título: {titulo}. "
        f"Objeto del contrato: {descripcion}. "
        f"Órgano: {organo}. "
        f"Lugar: {lugar_ejecucion}. "
        f"Plazo presentación: {plazo_ofertas}"
    )

    return {
        "titulo": titulo,
        "organo": organo,
        "fecha": fecha_str,
        "importe": importe,
        "enlace": enlace,
        "lugar": lugar_ejecucion,
        "plazo_ofertas": plazo_ofertas,
        "texto_completo": texto_evaluacion
    }


def cargar_cache():
    global GLOBAL_DF, GLOBAL_EMBEDDINGS

    try:
        if os.path.exists(CACHE_DB):
            GLOBAL_DF = pd.read_pickle(CACHE_DB)

        if os.path.exists(CACHE_EMB):
            GLOBAL_EMBEDDINGS = np.load(CACHE_EMB)

    except Exception as e:
        print(f"Error cargando caché: {e}")


cargar_cache()


@app.get("/", response_class=HTMLResponse)
def read_root():
    total_licitaciones = len(GLOBAL_DF)

    html_content = f"""<!DOCTYPE html>
<html lang="es">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Buscador Semántico de Licitaciones PLACSP</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="bg-gray-50 font-sans antialiased">

    <div class="max-w-7xl mx-auto px-4 py-8">

        <header class="mb-8 text-center">

            <h1 class="text-3xl font-extrabold text-blue-900">
                Buscador Semántico de Licitaciones Públicas (PLACSP)
            </h1>

            <p class="text-gray-600 mt-2">
                Búsqueda avanzada por Inteligencia Artificial, filtros por importe,
                fechas, órgano y sincronización con la plataforma.
            </p>

            <div
                id="status-badge"
                class="inline-block mt-3 px-4 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
            >
                Base de datos activa: {total_licitaciones} licitaciones
            </div>

        </header>


        <div class="grid grid-cols-1 lg:grid-cols-4 gap-8">


            <div class="bg-white p-6 rounded-xl shadow-md lg:col-span-1 space-y-4">

                <h2 class="text-lg font-bold text-gray-800 border-b pb-2">
                    Filtros de Búsqueda
                </h2>


                <div>

                    <label class="block text-sm font-medium text-gray-700">
                        Consulta Semántica
                    </label>

                    <input
                        type="text"
                        id="consulta"
                        placeholder="ej. mantenimiento informático..."
                        class="mt-1 w-full p-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                    >

                </div>


                <div>

                    <label class="block text-sm font-medium text-gray-700">
                        Órgano Contratante
                    </label>

                    <input
                        type="text"
                        id="organo"
                        placeholder="ej. Ayuntamiento..."
                        class="mt-1 w-full p-2 border border-gray-300 rounded-lg"
                    >

                </div>


                <div class="grid grid-cols-2 gap-2">

                    <div>

                        <label class="block text-xs font-medium text-gray-700">
                            Fecha Desde
                        </label>

                        <input
                            type="date"
                            id="fecha_desde"
                            class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm"
                        >

                    </div>


                    <div>

                        <label class="block text-xs font-medium text-gray-700">
                            Fecha Hasta
                        </label>

                        <input
                            type="date"
                            id="fecha_hasta"
                            class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm"
                        >

                    </div>

                </div>


                <div class="grid grid-cols-2 gap-2">

                    <div>

                        <label class="block text-xs font-medium text-gray-700">
                            Importe Mín (€)
                        </label>

                        <input
                            type="number"
                            id="importe_min"
                            value="0"
                            class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm"
                        >

                    </div>


                    <div>

                        <label class="block text-xs font-medium text-gray-700">
                            Importe Máx (€)
                        </label>

                        <input
                            type="number"
                            id="importe_max"
                            value="0"
                            class="mt-1 w-full p-2 border border-gray-300 rounded-lg text-sm"
                        >

                    </div>

                </div>


                <div>

                    <label class="block text-sm font-medium text-gray-700">
                        Umbral Similitud IA (%)
                    </label>

                    <input
                        type="range"
                        id="umbral"
                        min="50"
                        max="95"
                        value="70"
                        class="w-full mt-1"
                    >

                    <span
                        id="umbral-val"
                        class="text-xs text-gray-500"
                    >
                        70%
                    </span>

                </div>


                <button
                    onclick="realizarBusqueda()"
                    class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-200 shadow"
                >
                    Buscar Licitaciones
                </button>


                <button
                    onclick="buscarNovedades()"
                    class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-4 rounded-lg transition duration-200 shadow"
                >
                    Sincronizar Novedades
                </button>


                <button
                    onclick="exportarExcel()"
                    class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg transition duration-200 shadow"
                >
                    Descargar Excel
                </button>

            </div>



            <div class="bg-white p-6 rounded-xl shadow-md lg:col-span-3">

                <div class="flex justify-between items-center mb-4 border-b pb-2">

                    <h2 class="text-lg font-bold text-gray-800">
                        Resultados de Licitaciones
                    </h2>

                    <span
                        id="resultado-info"
                        class="text-sm text-gray-600"
                    >
                        Introduce una consulta o pulsa buscar.
                    </span>

                </div>


                <div
                    id="loading"
                    class="hidden text-center py-12"
                >

                    <div
                        class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"
                    ></div>

                    <p class="text-gray-600 mt-2">
                        Buscando licitaciones...
                    </p>

                </div>


                <div class="overflow-x-auto">

                    <table
                        id="tabla-resultados"
                        class="min-w-full divide-y divide-gray-200"
                    >

                        <thead class="bg-gray-100">

                            <tr>

                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                                    #
                                </th>

                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                                    Relevancia
                                </th>

                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                                    Título y Objeto
                                </th>

                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                                    Órgano
                                </th>

                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                                    Fecha
                                </th>

                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                                    Importe
                                </th>

                                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                                    Enlace
                                </th>

                            </tr>

                        </thead>


                        <tbody
                            class="bg-white divide-y divide-gray-200"
                            id="resultados-body"
                        >
                        </tbody>

                    </table>

                </div>

            </div>

        </div>

    </div>


    <script>

        document.getElementById('umbral').addEventListener('input', (e) => {

            document.getElementById('umbral-val').innerText =
                e.target.value + '%';

        });


        async function realizarBusqueda() {

            document.getElementById('loading')
                .classList.remove('hidden');

            const consulta =
                document.getElementById('consulta').value;

            const organo =
                document.getElementById('organo').value;

            const fecha_desde =
                document.getElementById('fecha_desde').value;

            const fecha_hasta =
                document.getElementById('fecha_hasta').value;

            const importe_min =
                document.getElementById('importe_min').value;

            const importe_max =
                document.getElementById('importe_max').value;

            const umbral =
                document.getElementById('umbral').value / 100;


            const response = await fetch('/api/buscar', {

                method: 'POST',

                headers: {
                    'Content-Type': 'application/json'
                },

                body: JSON.stringify({

                    consulta,
                    organo,
                    fecha_desde,
                    fecha_hasta,

                    importe_min:
                        parseFloat(importe_min),

                    importe_max:
                        parseFloat(importe_max),

                    umbral

                })

            });


            const data = await response.json();


            document.getElementById('loading')
                .classList.add('hidden');


            document.getElementById('resultado-info')
                .innerText = data.mensaje;


            const tbody =
                document.getElementById('resultados-body');

            tbody.innerHTML = '';


            data.resultados.forEach((row, index) => {

                const tr =
                    document.createElement('tr');


                tr.innerHTML = `

                    <td class="px-4 py-3 text-sm text-gray-500">
                        ${index + 1}
                    </td>

                    <td class="px-4 py-3 text-sm font-semibold text-blue-600">
                        ${row.relevancia}%
                    </td>

                    <td class="px-4 py-3 text-sm text-gray-900 font-medium">
                        ${row.titulo}
                    </td>

                    <td class="px-4 py-3 text-sm text-gray-600">
                        ${row.organo}
                    </td>

                    <td class="px-4 py-3 text-sm text-gray-600">
                        ${row.fecha}
                    </td>

                    <td class="px-4 py-3 text-sm text-gray-600">
                        ${row.importe.toLocaleString()} €
                    </td>

                    <td class="px-4 py-3 text-sm">

                        <a
                            href="${row.enlace}"
                            target="_blank"
                            class="text-blue-600 hover:underline font-bold"
                        >
                            Ver enlace
                        </a>

                    </td>

                `;


                tbody.appendChild(tr);

            });

        }


        async function buscarNovedades() {

            alert(
                "Conectando con PLACSP para descargar novedades. Esto puede tardar unos segundos..."
            );


            const response =
                await fetch('/api/actualizar', {
                    method: 'POST'
                });


            const data =
                await response.json();


            alert(data.mensaje);


            location.reload();

        }


        async function exportarExcel() {

            window.location.href =
                '/api/exportar';

        }

    </script>

</body>

</html>"""

    return html_content


@app.post("/api/buscar")
def api_buscar(payload: dict):

    global GLOBAL_DF, GLOBAL_EMBEDDINGS

    if GLOBAL_DF.empty:

        return {
            "resultados": [],
            "mensaje": "La base de datos está vacía. Sincroniza novedades."
        }


    df = GLOBAL_DF.copy()

    total_leidas = len(df)


    consulta = payload.get("consulta", "")

    organo_filtro = payload.get(
        "organo",
        ""
    ).lower()

    f_desde = payload.get(
        "fecha_desde",
        ""
    )

    f_hasta = payload.get(
        "fecha_hasta",
        ""
    )

    i_min = payload.get(
        "importe_min",
        0
    )

    i_max = payload.get(
        "importe_max",
        0
    )

    umbral = payload.get(
        "umbral",
        0.7
    )


    if f_desde:
        df = df[df["fecha"] >= f_desde]


    if f_hasta:
        df = df[df["fecha"] <= f_hasta]


    if i_min and i_min > 0:
        df = df[df["importe"] >= i_min]


    if i_max and i_max > 0:
        df = df[df["importe"] <= i_max]


    if organo_filtro:

        df = df[
            df["organo"]
            .str.lower()
            .str.contains(
                organo_filtro,
                na=False
            )
        ]


    if df.empty:

        return {
            "resultados": [],
            "mensaje": (
                f"No hay resultados con esos filtros "
                f"sobre {total_leidas} licitaciones."
            )
        }


    if consulta and consulta.strip():

        model = get_encoder()

        indices = df.index.tolist()


        if (
            GLOBAL_EMBEDDINGS is not None
            and len(GLOBAL_EMBEDDINGS) == len(GLOBAL_DF)
        ):

            embeddings_filtrados =
                GLOBAL_EMBEDDINGS[indices]

        else:

            embeddings_filtrados = model.encode(
                df["texto_completo"].tolist()
            )


        query_con_prefijo =
            f"query: {consulta.strip()}"


        embedding_query =
            model.encode(
                [query_con_prefijo]
            )


        similitudes =
            cosine_similarity(
                embedding_query,
                embeddings_filtrados
            )[0]


        df["Relevancia"] =
            np.round(
                similitudes * 100,
                2
            )


        df =
            df[
                df["Relevancia"]
                >= (umbral * 100)
            ]


        df =
            df.sort_values(
                "Relevancia",
                ascending=False
            )

    else:

        df["Relevancia"] = 100.0

        df =
            df.sort_values(
                "fecha",
                ascending=False
            )


    resultados = []


    for _, row in df.head(100).iterrows():

        resultados.append({

            "titulo":
                row.get(
                    "titulo",
                    ""
                ),

            "organo":
                row.get(
                    "organo",
                    ""
                ),

            "fecha":
                row.get(
                    "fecha",
                    ""
                ),

            "importe":
                row.get(
                    "importe",
                    0.0
                ),

            "enlace":
                row.get(
                    "enlace",
                    ""
                ),

            "relevancia":
                row.get(
                    "Relevancia",
                    100.0
                )

        })


    return {

        "resultados":
            resultados,

        "mensaje":
            (
                f"Se encontraron "
                f"{len(resultados)} "
                f"licitaciones "
                f"(analizadas "
                f"{total_leidas})."
            )

    }


@app.post("/api/actualizar")
def api_actualizar():

    global GLOBAL_DF, GLOBAL_EMBEDDINGS

    model = get_encoder()


    enlaces_conocidos = (
        set(GLOBAL_DF["enlace"])
        if not GLOBAL_DF.empty
        else set()
    )


    nuevas_totales = []


    headers = {
        'User-Agent':
            'Mozilla/5.0'
    }


    for fuente in FUENTES:

        try:

            url = fuente["url"]


            resp = requests.get(
                url,
                headers=headers,
                timeout=30
            )


            if resp.status_code == 200:

                root =
                    ET.fromstring(
                        resp.content
                    )


                for entry in root.findall(
                    "atom:entry",
                    NS
                ):

                    datos =
                        parsear_entrada_xml(
                            entry
                        )


                    if (
                        datos["enlace"]
                        not in enlaces_conocidos
                    ):

                        nuevas_totales.append(
                            datos
                        )

                        enlaces_conocidos.add(
                            datos["enlace"]
                        )


        except Exception as e:

            print(
                f"Error en fuente: {e}"
            )


    if not nuevas_totales:

        return {
            "mensaje":
                "La base de datos ya está al día. No hay novedades."
        }


    df_nuevas =
        pd.DataFrame(
            nuevas_totales
        )


    nuevos_embeddings =
        model.encode(
            df_nuevas[
                "texto_completo"
            ].tolist(),
            show_progress_bar=False
        )


    if GLOBAL_DF.empty:

        GLOBAL_DF =
            df_nuevas

        GLOBAL_EMBEDDINGS =
            nuevos_embeddings

    else:

        GLOBAL_DF =
            pd.concat(
                [
                    df_nuevas,
                    GLOBAL_DF
                ],
                ignore_index=True
            )


        GLOBAL_EMBEDDINGS =
            np.vstack(
                [
                    nuevos_embeddings,
                    GLOBAL_EMBEDDINGS
                ]
            )


    try:

        GLOBAL_DF.to_pickle(
            CACHE_DB
        )

        np.save(
            CACHE_EMB,
            GLOBAL_EMBEDDINGS
        )

    except Exception:

        pass


    return {

        "mensaje":
            (
                f"¡Actualización exitosa! "
                f"Se agregaron "
                f"+{len(df_nuevas)} "
                f"licitaciones nuevas."
            )

    }


@app.get("/api/exportar")
def api_exportar():

    global GLOBAL_DF


    if GLOBAL_DF.empty:

        return {
            "error":
                "No hay datos para exportar"
        }


    ruta_excel =
        os.path.join(
            DATA_DIR,
            "licitaciones_exportadas.xlsx"
        )


    GLOBAL_DF.to_excel(
        ruta_excel,
        index=False,
        engine='openpyxl'
    )


    return FileResponse(
        ruta_excel,
        filename="licitaciones.xlsx"
    )


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
