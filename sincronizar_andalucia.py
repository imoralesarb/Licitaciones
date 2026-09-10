# Sincronización de licitaciones de Andalucía con Supabase
# Incluye la función construir_enlace y la verificación de orden descendente.

from datetime import datetime, date
import os
import time
import re
import requests

from bs4 import BeautifulSoup
from supabase import create_client, Client

# ========================================================
# CONFIGURACIÓN SUPABASE Y GENERAL
# ========================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

FUENTE_ANDALUCIA = "Andalucía"
TAMANO_LOTE = 50

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
}


# ========================================================
# FUNCIONES AUXILIARES Y DE ENLACES
# ========================================================

def construir_enlace(expediente_id):
    """
    Construye la URL oficial de detalle para una licitación de Andalucía.
    """
    if not expediente_id:
        return ""
    return f"https://www.juntadeandalucia.es/contratacion/consultas/detalle?idExpediente={expediente_id}"


def verificar_orden_descendente(lista_licitaciones):
    """
    Comprueba que la lista de licitaciones esté ordenada de más reciente 
    a más antigua según la fecha de publicación.
    """
    if not lista_licitaciones or len(lista_licitaciones) < 2:
        print("[VERIFICACIÓN] La lista es demasiado corta para validar el orden.")
        return True

    ordenado = True
    for i in range(len(lista_licitaciones) - 1):
        fecha_actual = lista_licitaciones[i].get("fecha_publicacion")
        fecha_siguiente = lista_licitaciones[i+1].get("fecha_publicacion")

        if fecha_actual and fecha_siguiente:
            if fecha_actual < fecha_siguiente:
                print(f"[ALERTA DE ORDEN] Desorden en índice {i}: {fecha_actual} es anterior a {fecha_siguiente}")
                ordenado = False

    if ordenado:
        print("[VERIFICACIÓN] Las licitaciones están correctamente ordenadas de forma descendente.")
    else:
        print("[VERIFICACIÓN] ¡Atención! Se han detectado elementos fuera de orden.")
    
    return ordenado


# ========================================================
# EXTRACCIÓN Y PROCESAMIENTO DE DATOS
# ========================================================

def extraer_datos_detalle(expediente_id):
    """
    Función que procesa los datos detallados de una licitación utilizando el expediente_id.
    """
    enlace = construir_enlace(expediente_id)
    # Lógica de extracción adicional...
    return enlace


def main():
    print("======================================================================")
    print("SINCRONIZACIÓN DE LICITACIONES - ANDALUCÍA")
    print("======================================================================")
    
    # Simulación de recuperación de licitaciones para el script completo
    # (Aquí iría la llamada al buscador de Andalucía y la carga inicial)
    fecha_hoy = date.today().strftime("%d/%m/%Y")
    print(f"Fecha actual: {fecha_hoy}")
    
    # Ejemplo de lista obtenida
    licitaciones_recuperadas = [
        {"expediente": "403648", "fecha_publicacion": "2026-09-10"},
        {"expediente": "403649", "fecha_publicacion": "2026-09-09"}
    ]
    
    # Comprobación de orden
    verificar_orden_descendente(licitaciones_recuperadas)
    
    print("======================================================================")
    print("PROCESANDO LICITACIONES")
    print("======================================================================")
    
    for idx, reg in enumerate(licitaciones_recuperadas, start=1):
        exp_id = reg.get("expediente")
        print(f"[{idx}/{len(licitaciones_recuperadas)}] Expediente: {exp_id}")
        enlace = construir_enlace(exp_id)
        print(f"    enlace = {enlace}")


if __name__ == "__main__":
    main()
