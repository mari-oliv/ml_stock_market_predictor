from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_root() -> None:
    """Valida que o endpoint raiz responde com status 200 e payload esperado."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "api is working"}


def test_predict_validation_error_when_missing_symbol() -> None:
    """Valida que o /predict exige o campo `symbol`."""
    client = TestClient(app)
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_success(monkeypatch) -> None:
    """Valida /predict com mocks para não depender de TensorFlow/DB."""
    import src.api.routes as routes

    def _fake_predict_next_price(**_kwargs):
        return 123.456

    def _fake_save_prediction(_symbol: str, _value: float, _date_iso: str) -> None:
        return None

    monkeypatch.setattr(routes, "predict_next_price", _fake_predict_next_price)
    monkeypatch.setattr(routes, "save_prediction", _fake_save_prediction)

    client = TestClient(app)
    response = client.post("/predict", json={"symbol": "AAPL"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["value"] == 123.46
    assert isinstance(payload["date"], str)
    assert payload["date"]