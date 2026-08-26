from fastapi import FastAPI
import os
import pandas as pd

app = FastAPI(title="Buscador de Licitaciones PLACSP", version="2.3")

DATA_DIR = "/tmp/licitaciones_data"
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_DB = os.path.join(DATA_DIR, "licitaciones_db.pkl")

# Inicializamos un DataFrame global seguro
GLOBAL_DF = pd.DataFrame(columns=["titulo", "organo", "fecha", "importe", "enlace"])

def cargar_cache():
    global GLOBAL_DF
    try:
        if os.path.exists(CACHE_DB):
            temp_df = pd.read_pickle(CACHE_DB)
            if not temp_df.empty:
                GLOBAL_DF = temp_df
    except Exception as e:
        print(f"Aviso al cargar caché (normal en primer arranque): {e}")

cargar_cache()

# Importamos y registramos las rutas desde nuestro archivo auxiliar
try:
    from logica import registrar_rutas
    registrar_rutas(app, GLOBAL_DF, DATA_DIR, CACHE_DB)
except Exception as e:
    print(f"Error al registrar rutas: {e}")
