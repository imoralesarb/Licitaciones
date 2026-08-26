from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd

app = FastAPI(title="Buscador de Licitaciones PLACSP", version="2.2")
