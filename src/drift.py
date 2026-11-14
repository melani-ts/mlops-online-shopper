"""Compara los datos del entrenamiento con los que llegaron a la API y guarda un informe de drift.

Uso:  python src/drift.py   compara con logs/actual.csv
Escribe reports/drift.html, que se abre en el navegador.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

ACTUAL = sys.argv[1] if len(sys.argv) > 1 else "logs/actual.csv"

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["prepare"]
referencia = pd.read_csv("data/raw/online_shoppers_intention.csv").drop(columns=[params["target"], *params.get("drop_columns", [])])
actual = pd.read_csv(ACTUAL)

# Solo se comparan las columnas que están en los dos archivos, en el mismo orden.
columnas = [c for c in referencia.columns if c in actual.columns]
referencia, actual = referencia[columnas], actual[columnas]

informe = Report([DataDriftPreset()]).run(reference_data=referencia, current_data=actual)
Path("reports").mkdir(exist_ok=True)
informe.save_html("reports/drift.html")

cambiaron = []
for metrica in informe.dict()["metrics"]:
    nombre, valor = metrica["metric_name"], metrica["value"]
    if nombre.startswith("DriftedColumnsCount"):
        parte = valor["share"]
    elif nombre.startswith("ValueDrift"):
        columna = metrica["config"]["column"]
        umbral = float(nombre.split("threshold=")[1].rstrip(")"))
        # Con p-valor, hay cambio si es menor que el umbral; con distancias, si es mayor o igual.
        hay_cambio = valor < umbral if "p_value" in nombre else valor >= umbral
        if hay_cambio:
            cambiaron.append(columna)

print(f"Columnas con drift: {len(cambiaron)} de {len(columnas)} ({parte:.0%}): {', '.join(cambiaron) or 'ninguna'}")
print("Hay drift en el conjunto." if parte >= 0.5 else "El conjunto no tiene drift: cambiaron menos de la mitad de las columnas.")
print("Informe: reports/drift.html")
