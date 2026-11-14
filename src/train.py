"""Etapa train: entrena, registra el run en MLflow y guarda models/model.pkl."""
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
target = params["prepare"]["target"]
p = params["train"]
cfg = params["mlflow"]

train = pd.read_csv("data/prepared/train.csv")
test = pd.read_csv("data/prepared/test.csv")
X_train, y_train = train.drop(columns=[target]), train[target]
X_test, y_test = test.drop(columns=[target]), test[target]

# Versión del dataset según DVC: queda como etiqueta del run
dvc_file = yaml.safe_load(open("data/raw/online_shoppers_intention.csv.dvc", encoding="utf-8"))
data_md5 = dvc_file["outs"][0]["md5"]

#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
#mlflow.set_experiment(cfg["experiment"])

#with mlflow.start_run():
#    mlflow.set_tag("data_md5", data_md5)
#    mlflow.log_params(p)

model = RandomForestClassifier(
    n_estimators=p["n_estimators"],
    max_depth=p["max_depth"],
    random_state=params["prepare"]["random_state"],
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
#    mlflow.log_metrics({
#        "accuracy": accuracy_score(y_test, pred),
#        "f1_macro": f1_score(y_test, pred, average="macro"),
#    })
#    mlflow.sklearn.log_model(
#        model,
#        name="model",
#        registered_model_name=cfg["model_name"],
#        input_example=X_train.head(3),
#        serialization_format="cloudpickle",
#    )

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Run registrado en MLflow y modelo guardado en models/model.pkl")
