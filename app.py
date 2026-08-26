from fastapi import FastAPI
import os
import pandas as pd

app = FastAPI(title="Buscador de Licitaciones PLACSP", version="2.2")

DATA_DIR = "/tmp/licitaciones_data"
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_DB = os.path.join(DATA_DIR, "licitaciones_db.pkl")

GLOBAL_DF = pd.DataFrame()

def cargar_cache():
    global GLOBAL_DF
    try:
        if os.path.exists(CACHE_DB):
            GLOBAL_DF = pd.read_pickle(CACHE_DB)
    except Exception as e:
        print(f"Error cargando caché: {e}")

cargar_cache()

# Importamos y registramos las rutas desde nuestro archivo auxiliar
from logica import registrar_rutas
registrar_rutas(app, GLOBAL_DF, DATA_DIR, CACHE_DB)
