import json
import random
from pathlib import Path
import torch
from config import CARPETA_DATOS, CARPETA_MODELOS, CARPETA_RESULTADOS, SEMILLA

def preparar_carpetas():
    for carpeta in [CARPETA_DATOS, CARPETA_MODELOS, CARPETA_RESULTADOS]:
        carpeta.mkdir(parents=True, exist_ok=True)

def fijar_semilla():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMILLA)

def guardar_json(datos, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

def cargar_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
