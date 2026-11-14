"""Simula los datos que llegarían a la API después de un cambio en el mundo, como una campaña nueva.

Uso:  python src/simular_drift.py
Escribe logs/actual.csv con filas del dataset original en las que cambian las columnas de CAMBIOS.

Cambien CAMBIOS por columnas de su dataset:
- Columna numérica: el número multiplica la columna. 0.3 la deja en un 30 %; 1.5 la sube un 50 %.
- Columna de texto: el valor reemplaza a esa categoría en el 60 % de las filas.
"""
from pathlib import Path

import pandas as pd
import yaml

# Simulación: una campaña en redes trae muchos visitantes nuevos que ven menos páginas de valor.
CAMBIOS = {
    "PageValues": 0.3,
    "VisitorType": "New_Visitor",
}
FILAS = 500

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["prepare"]
datos = pd.read_csv("data/raw/online_shoppers_intention.csv").drop(columns=[params["target"], *params.get("drop_columns", [])])
actual = datos.sample(FILAS, random_state=1).reset_index(drop=True)

for columna, cambio in CAMBIOS.items():
    if columna not in actual.columns:
        raise SystemExit(f"La columna {columna} no está en el dataset. Revisen CAMBIOS.")
    if isinstance(cambio, str):
        filas = actual.sample(frac=0.6, random_state=2).index
        actual.loc[filas, columna] = cambio
    else:
        nuevo = actual[columna] * cambio
        actual[columna] = nuevo.round().astype(int) if pd.api.types.is_integer_dtype(actual[columna]) else nuevo.round(2)

Path("logs").mkdir(exist_ok=True)
actual.to_csv("logs/actual.csv", index=False)
print(f"logs/actual.csv: {len(actual)} filas. Columnas cambiadas: {', '.join(CAMBIOS)}")
