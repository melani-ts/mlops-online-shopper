import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(_file_).resolve().parents[1]))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    X = pd.DataFrame({"tenure": [1, 60], "MonthlyCharges": [90, 20]})
    joblib.dump(LogisticRegression().fit(X, ["Yes", "No"]), tmp_path / "model.pkl")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import api.app as modulo
    importlib.reload(modulo)
    return TestClient(modulo.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json={"tenure": 2})
    assert r.status_code == 200
    assert r.json()["prediccion"] in {"Yes", "No"}


def test_dato_invalido(client):
    assert client.post("/predict", json={"tenure": "muchos"}).status_code == 422
