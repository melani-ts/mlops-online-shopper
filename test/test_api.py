import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    X = pd.DataFrame({"ProductRelated_Duration": [0, 100], "BounceRates": [0, 1]})
    joblib.dump(LogisticRegression().fit(X, [False, True]), tmp_path / "model.pkl")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import api.app as modulo
    importlib.reload(modulo)
    return TestClient(modulo.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_dato_invalido(client):
    assert client.post("/predict", json={"ProductRelated_Duration": "invalid", "BounceRates": 0.9}).status_code == 422