"""API de inferencia: carga el modelo y responde si una sesión de la tienda en línea terminará en compra (Revenue)."""
import csv
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import REGISTRY, Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

LOGS = Path(os.getenv("LOGS_DIR", "logs"))
modelo = joblib.load(Path(os.getenv("MODEL_DIR", "models")) / "model.pkl")
app = FastAPI(title="API de predicción de Churn (Telco)")
Instrumentator().instrument(app).expose(app)

try:
    PREDICCIONES = Counter("predicciones_total", "Predicciones por clase", ["clase"])
except ValueError:  # ya existe: pasa cuando las pruebas vuelven a cargar la API
    PREDICCIONES = REGISTRY._names_to_collectors["predicciones_total"]


def guardar_entrada(datos):
    """Agrega una fila a logs/entradas.csv con lo que recibió la API."""
    LOGS.mkdir(exist_ok=True)
    archivo = LOGS / "entradas.csv"
    nuevo = not archivo.exists()
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(datos))
        if nuevo:
            escritor.writeheader()
        escritor.writerow(datos)

class Sesion(BaseModel):
    """Columnas del dataset (sin Revenue). Los valores por defecto son la mediana o la moda del dataset."""

    Administrative: int = Field(1, ge=0, description="Páginas administrativas visitadas")
    Administrative_Duration: float = Field(7.5, ge=0, description="Segundos en páginas administrativas")
    Informational: int = Field(0, ge=0, description="Páginas informativas visitadas")
    Informational_Duration: float = Field(0.0, ge=0, description="Segundos en páginas informativas")
    ProductRelated: int = Field(18, ge=0, description="Páginas de producto visitadas")
    ProductRelated_Duration: float = Field(598.94, ge=0, description="Segundos en páginas de producto")
    BounceRates: float = Field(0.0031, ge=0, le=1, description="Tasa de rebote promedio")
    ExitRates: float = Field(0.0252, ge=0, le=1, description="Tasa de salida promedio")
    PageValues: float = Field(0.0, ge=0, description="Valor promedio de las páginas visitadas")
    SpecialDay: float = Field(0.0, ge=0, le=1, description="Cercanía a una fecha especial (0 a 1)")
    Month: Literal["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] = "May"
    OperatingSystems: int = Field(2, ge=1, description="Sistema operativo (código)")
    Browser: int = Field(2, ge=1, description="Navegador (código)")
    Region: int = Field(3, ge=1, description="Región (código)")
    TrafficType: int = Field(2, ge=1, description="Fuente de tráfico (código)")
    VisitorType: Literal["New_Visitor", "Returning_Visitor", "Other"] = "Returning_Visitor"
    Weekend: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(sesion: Sesion):
    guardar_entrada(sesion.model_dump())
    X = pd.get_dummies(pd.DataFrame([sesion.model_dump()]), dtype=int)
    X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)
    resultado = str(modelo.predict(X)[0])
    PREDICCIONES.labels(clase=resultado).inc()
    return {"prediccion": resultado}
