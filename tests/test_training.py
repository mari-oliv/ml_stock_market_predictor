from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_train_starts_and_check_train_returns_all_fields(monkeypatch) -> None:
    """Valida que /train dispara treino e /check_train retorna status + novos campos."""
    import src.api.routes as routes

    def _fake_collect_prereqs():
        return {
            "checked_at": "2026-01-09 00:00:00 -03:00",
            "kernel_name": "python3",
            "kernel_available": True,
            "missing_modules": [],
            "ok": True,
        }

    def _fake_run_notebook() -> None:
        with routes._TRAIN_LOCK:
            routes.TRAIN_STATE["status"] = "running"
            routes.TRAIN_STATE["phase"] = "execute"
            routes.TRAIN_STATE["started_at"] = "2026-01-09 00:00:00 -03:00"
            routes.TRAIN_STATE["execute_engine"] = "papermill"
            routes.TRAIN_STATE["execute_started_at"] = "2026-01-09 00:00:01 -03:00"
            routes.TRAIN_STATE["papermill_started_at"] = "2026-01-09 00:00:01 -03:00"
            routes.TRAIN_STATE["nbclient_started_at"] = None
            routes.TRAIN_STATE["metrics"] = {
                "engine": "papermill",
                "find_notebook_s": 0.001,
                "prepare_paths_s": 0.001,
                "execute_s": 0.001,
                "notebook_in": routes.TRAIN_STATE.get("notebook"),
                "notebook_out": "/tmp/out.ipynb",
                "kernel_name": "python3",
                "kernel_startup_timeout_s": 900,
            }

        with routes._TRAIN_LOCK:
            routes.TRAIN_STATE["status"] = "succeeded"
            routes.TRAIN_STATE["phase"] = "done"
            routes.TRAIN_STATE["ended_at"] = "2026-01-09 00:00:02 -03:00"
            routes.TRAIN_STATE["duration_s"] = 2.0
            routes.TRAIN_STATE["last_output_ipynb"] = "/tmp/out.ipynb"
            routes.TRAIN_STATE["error"] = None

    class _ImmediateThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self):
            self._target()

    def _fake_save_metrics_snapshot(**_kwargs):
        return None

    monkeypatch.setattr(routes, "_collect_train_prereqs", _fake_collect_prereqs)
    monkeypatch.setattr(routes, "_run_notebook", _fake_run_notebook)
    monkeypatch.setattr(routes.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(routes, "save_train_metrics_snapshot", _fake_save_metrics_snapshot)

    client = TestClient(app)
    train_resp = client.post("/train")
    assert train_resp.status_code == 200
    train_payload = train_resp.json()
    assert train_payload["status"] in {"started", "running"}
    assert train_payload.get("notebook")

    check_resp = client.get("/check_train")
    assert check_resp.status_code == 200
    payload = check_resp.json()

    assert payload["train_status"] in {"running", "succeeded", "failed", "idle"}
    assert "phase" in payload
    assert "prereqs" in payload
    assert "execute_engine" in payload
    assert "execute_started_at" in payload
    assert "papermill_started_at" in payload
    assert "nbclient_started_at" in payload

    assert "metrics" in payload
    assert "artifact_found" in payload
    assert "artifact_path" in payload