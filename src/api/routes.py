import glob
import importlib
import logging
import os
import threading
from datetime import datetime
from time import perf_counter
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.inference.predict_next import _resolve_artifact_path, predict_next_price
from src.shared.predictions_store import save_prediction, save_train_metrics_snapshot
from src.utils.datetime_utils import brasilia_iso, brasilia_now
from src.utils.logging_config import setup_logging
from src.utils.timer import section

setup_logging()
logger = logging.getLogger("app.api")

TRAIN_KERNEL_NAME = os.getenv("TRAIN_KERNEL_NAME", "python3")
TRAIN_KERNEL_STARTUP_TIMEOUT_S = int(float(os.getenv("TRAIN_KERNEL_STARTUP_TIMEOUT_S", "900")))

router = APIRouter()


class PredictRequest(BaseModel):
    """Payload de entrada para o endpoint de predição.

    `data_source="auto"` tenta yfinance primeiro e, se falhar, usa CSV.
    """

    symbol: str
    data_source: Literal["auto", "yfinance", "csv"] = "auto"
    csv_path: Optional[str] = None
    csv_target_col: Optional[str] = None
    artifact_path: Optional[str] = None


@router.get("/")
def health():
    """Endpoint de verificação de saúde da API."""
    return {"status": "api is working"}


@router.post("/predict")
def predict(req: PredictRequest):
    """Gera a próxima predição de preço e persiste o resultado quando possível."""
    extra = {
        "symbol": req.symbol,
        "data_source": req.data_source,
        "csv_path": bool(req.csv_path),
        "artifact_path": bool(req.artifact_path),
    }
    with section("predict", logger, extra):
        try:
            value, data_source_used = predict_next_price(
                symbol=req.symbol,
                artifact_path=req.artifact_path,
                data_source=req.data_source,
                csv_path=req.csv_path,
                csv_target_col=req.csv_target_col,
            )
            now_iso = brasilia_iso()
            try:
                save_prediction(req.symbol, value, now_iso)
            except Exception as e:
                logger.warning(f"predict:save_prediction error={e}")
            return {
                "value": round(value, 2),
                "symbol": req.symbol,
                "date": now_iso,
                "data_source_requested": req.data_source,
                "data_source_used": data_source_used,
            }
        except Exception as e:
            logger.exception(f"predict:error symbol={req.symbol}")
            raise HTTPException(status_code=500, detail=str(e))


TRAIN_STATE = {
    "status": "idle",
    "phase": "idle",
    "started_at": None,
    "ended_at": None,
    "duration_s": None,
    "error": None,
    "last_output_ipynb": None,
    "notebook": None,
    "prereqs": None,
    "execute_engine": None,
    "execute_started_at": None,
    "nbclient_started_at": None,
    "metrics": {},
    "metrics_saved": False,
}
_TRAIN_LOCK = threading.Lock()


def _parse_datetime(value: str) -> datetime:
    """Faz parse de timestamps com e sem espaço antes do offset.

    Aceita:
    - `YYYY-MM-DDTHH:MM:SS-03:00`
    - `YYYY-MM-DD HH:MM:SS-03:00`
    - `YYYY-MM-DD HH:MM:SS -03:00`
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        parts = value.split(" ")
        if len(parts) == 3:
            return datetime.fromisoformat(f"{parts[0]} {parts[1]}{parts[2]}")
        raise


def _project_root() -> str:
    """Retorna o caminho absoluto da raiz do projeto a partir deste módulo."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def _collect_train_prereqs() -> dict[str, Any]:
    """Coleta informações sobre kernel e dependências necessárias para o treino.

    Isso ajuda a validar se o kernel está disponível e se as libs do runner existem no ambiente.
    """
    required = ["nbclient", "nbformat", "ipykernel", "jupyter_client"]
    missing: list[str] = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)

    kernel_available: Optional[bool] = None
    if "jupyter_client" not in missing:
        try:
            from jupyter_client.kernelspec import KernelSpecManager

            specs = KernelSpecManager().find_kernel_specs()
            kernel_available = TRAIN_KERNEL_NAME in specs
        except Exception:
            kernel_available = None

    return {
        "checked_at": brasilia_iso(),
        "kernel_name": TRAIN_KERNEL_NAME,
        "kernel_available": kernel_available,
        "missing_modules": missing,
        "ok": (len(missing) == 0) and (kernel_available is True),
    }


def _find_notebook(root: str) -> str:
    """Localiza o notebook de treino, usando env vars e caminhos padrão."""
    env_nb = os.getenv("TRAIN_NOTEBOOK") or os.getenv("TRAIN_NOTEBOOK_PATH")
    candidates = []
    if env_nb:
        candidates.append(env_nb)

    candidates.append(os.path.join(root, "notebooks", "notebook.ipynb"))
    candidates.append(
        os.path.abspath(
            os.path.join(root, "..", "tc4", "ml-unified-service", "notebooks", "notebook.ipynb")
        )
    )

    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)

    for base in [root, os.path.abspath(os.path.join(root, ".."))]:
        matches = glob.glob(os.path.join(base, "**", "notebooks", "notebook.ipynb"), recursive=True)
        if matches:
            return os.path.abspath(matches[0])

    raise FileNotFoundError(
        "Notebook não encontrado. Defina TRAIN_NOTEBOOK com o caminho completo do notebook.ipynb."
    )


def _run_notebook():
    """Executa o notebook de treino de forma assíncrona e atualiza o estado global."""
    with _TRAIN_LOCK:
        TRAIN_STATE["status"] = "running"
        TRAIN_STATE["phase"] = "starting"
        TRAIN_STATE["started_at"] = brasilia_iso()
        TRAIN_STATE["ended_at"] = None
        TRAIN_STATE["duration_s"] = None
        TRAIN_STATE["error"] = None
        TRAIN_STATE["last_output_ipynb"] = None
        TRAIN_STATE["metrics_saved"] = False
        TRAIN_STATE["execute_engine"] = None
        TRAIN_STATE["execute_started_at"] = None
        TRAIN_STATE["nbclient_started_at"] = None
        TRAIN_STATE["metrics"] = {
            "engine": None,
            "find_notebook_s": None,
            "prepare_paths_s": None,
            "execute_s": None,
            "notebook_in": None,
            "notebook_out": None,
            "kernel_name": TRAIN_KERNEL_NAME,
            "kernel_startup_timeout_s": TRAIN_KERNEL_STARTUP_TIMEOUT_S,
        }

    root = _project_root()
    logger.info(f"train:triggered root={root}")

    try:
        prereqs = _collect_train_prereqs()
        with _TRAIN_LOCK:
            TRAIN_STATE["prereqs"] = prereqs
            TRAIN_STATE["phase"] = "find_notebook"

        t0 = perf_counter()
        with section("train.find_notebook", logger, {"root": root}):
            nb_in = _find_notebook(root)
        find_s = round((perf_counter() - t0), 3)
        with _TRAIN_LOCK:
            TRAIN_STATE["notebook"] = nb_in
            TRAIN_STATE["metrics"]["notebook_in"] = nb_in
            TRAIN_STATE["metrics"]["find_notebook_s"] = find_s
        logger.info(f"train:notebook path={nb_in}")

        t0 = perf_counter()
        with _TRAIN_LOCK:
            TRAIN_STATE["phase"] = "prepare_paths"
        with section("train.prepare_paths", logger):
            nb_dir = os.path.dirname(nb_in)
            out_dir = os.path.join(nb_dir, "runs")
            os.makedirs(out_dir, exist_ok=True)
            ts = brasilia_now().strftime("%Y%m%d-%H%M%S")
            base = os.path.splitext(os.path.basename(nb_in))[0]
            nb_out = os.path.join(out_dir, f"{base}-run-{ts}.ipynb")
        prep_s = round((perf_counter() - t0), 3)
        with _TRAIN_LOCK:
            TRAIN_STATE["metrics"]["prepare_paths_s"] = prep_s
            TRAIN_STATE["metrics"]["notebook_out"] = nb_out

        t0 = perf_counter()
        import nbformat
        from nbclient import NotebookClient

        with _TRAIN_LOCK:
            TRAIN_STATE["phase"] = "execute"
            TRAIN_STATE["execute_engine"] = "nbclient"
            TRAIN_STATE["execute_started_at"] = brasilia_iso()
            TRAIN_STATE["nbclient_started_at"] = TRAIN_STATE["execute_started_at"]
        with section("train.execute", logger, {"engine": "nbclient", "nb_out": nb_out}):
            nb = nbformat.read(nb_in, as_version=4)
            client = NotebookClient(
                nb,
                timeout=None,
                startup_timeout=TRAIN_KERNEL_STARTUP_TIMEOUT_S,
                kernel_name=TRAIN_KERNEL_NAME,
                allow_errors=False,
            )
            client.execute()
            nbformat.write(nb, nb_out)
        engine_used = "nbclient"

        exec_s = round((perf_counter() - t0), 3)
        with _TRAIN_LOCK:
            TRAIN_STATE["metrics"]["engine"] = engine_used
            TRAIN_STATE["metrics"]["execute_s"] = exec_s

        with _TRAIN_LOCK:
            TRAIN_STATE["status"] = "succeeded"
            TRAIN_STATE["phase"] = "finalizing"
            TRAIN_STATE["last_output_ipynb"] = nb_out
        logger.info(f"train:succeeded output={nb_out}")
    except Exception as ex:
        with _TRAIN_LOCK:
            TRAIN_STATE["status"] = "failed"
            TRAIN_STATE["phase"] = "finalizing"
            TRAIN_STATE["error"] = str(ex)
        logger.exception("train:failed")
    finally:
        with _TRAIN_LOCK:
            TRAIN_STATE["ended_at"] = brasilia_iso()
            TRAIN_STATE["phase"] = "done"
            try:
                if TRAIN_STATE["started_at"] and TRAIN_STATE["ended_at"]:
                    start_dt = _parse_datetime(TRAIN_STATE["started_at"])
                    end_dt = _parse_datetime(TRAIN_STATE["ended_at"])
                    TRAIN_STATE["duration_s"] = round((end_dt - start_dt).total_seconds(), 3)
            except Exception:
                TRAIN_STATE["duration_s"] = None

            status_snapshot = TRAIN_STATE.get("status")
            started_at_snapshot = TRAIN_STATE.get("started_at")
            ended_at_snapshot = TRAIN_STATE.get("ended_at")
            duration_s_snapshot = TRAIN_STATE.get("duration_s")
            error_snapshot = TRAIN_STATE.get("error")
            notebook_snapshot = TRAIN_STATE.get("notebook")
            last_output_ipynb_snapshot = TRAIN_STATE.get("last_output_ipynb")
            metrics_snapshot = (TRAIN_STATE.get("metrics") or {}).copy()
            metrics_saved_snapshot = bool(TRAIN_STATE.get("metrics_saved"))

        if status_snapshot in {"succeeded", "failed"} and not metrics_saved_snapshot:
            try:
                save_train_metrics_snapshot(
                    train_status=status_snapshot,
                    started_at=started_at_snapshot,
                    ended_at=ended_at_snapshot,
                    duration_s=duration_s_snapshot,
                    elapsed_s=duration_s_snapshot,
                    notebook=notebook_snapshot,
                    last_output_ipynb=last_output_ipynb_snapshot,
                    metrics=metrics_snapshot,
                    error=error_snapshot,
                )
                with _TRAIN_LOCK:
                    TRAIN_STATE["metrics_saved"] = True
            except Exception:
                pass

        logger.info(
            f"train:end status={TRAIN_STATE['status']} duration_s={TRAIN_STATE.get('duration_s')} "
            f"output={TRAIN_STATE.get('last_output_ipynb')}"
        )


@router.post("/train")
def start_train(notebook: Optional[str] = None):
    """Dispara o treino em background executando um notebook (com override opcional)."""
    if notebook:
        nb_path = os.path.abspath(os.path.expanduser(notebook))
        if not os.path.exists(nb_path):
            logger.warning(f"train:override_notebook not_found path={nb_path}")
            raise HTTPException(status_code=400, detail=f"Notebook não encontrado em: {nb_path}")
        os.environ["TRAIN_NOTEBOOK"] = nb_path
        logger.info(f"train:override_notebook path={nb_path}")

    try:
        root = _project_root()
        resolved_nb = _find_notebook(root)
        with _TRAIN_LOCK:
            TRAIN_STATE["notebook"] = resolved_nb
    except Exception as e:
        logger.warning(f"train:notebook_not_found error={e}")
        raise HTTPException(status_code=400, detail=str(e))

    if TRAIN_STATE["status"] == "running":
        logger.info("train:already_running")
        return {"status": "running", "started_at": TRAIN_STATE["started_at"]}

    t = threading.Thread(target=_run_notebook, daemon=True)
    t.start()
    now = brasilia_iso()
    logger.info("train:started")
    return {
        "status": "started",
        "started_at": now,
        "notebook": resolved_nb,
        "message": "Treino do modelo iniciado, aguarde em torno de 15 à 30 minutos para verificar o status",
    }


@router.get("/check_train")
def check_train(symbol: Optional[str] = None):
    """Retorna o status do treino e informa se há artefato disponível para o símbolo."""
    prereqs_snapshot = None
    with _TRAIN_LOCK:
        if TRAIN_STATE.get("prereqs") is not None:
            prereqs_snapshot = TRAIN_STATE.get("prereqs")

    if prereqs_snapshot is None:
        try:
            prereqs_snapshot = _collect_train_prereqs()
            with _TRAIN_LOCK:
                TRAIN_STATE["prereqs"] = prereqs_snapshot
        except Exception:
            prereqs_snapshot = None

    with _TRAIN_LOCK:
        train_status = TRAIN_STATE["status"]
        phase_state = TRAIN_STATE.get("phase")
        started_at_state = TRAIN_STATE.get("started_at")
        ended_at_state = TRAIN_STATE.get("ended_at")
        duration_s_state = TRAIN_STATE.get("duration_s")
        error_state = TRAIN_STATE.get("error")
        notebook_state = TRAIN_STATE.get("notebook")
        last_output_ipynb_state = TRAIN_STATE.get("last_output_ipynb")
        metrics_state = TRAIN_STATE.get("metrics", {})
        metrics_saved_state = bool(TRAIN_STATE.get("metrics_saved"))
        execute_engine_state = TRAIN_STATE.get("execute_engine")
        execute_started_at_state = TRAIN_STATE.get("execute_started_at")
        nbclient_started_at_state = TRAIN_STATE.get("nbclient_started_at")

    elapsed_s = None
    try:
        if ended_at_state and duration_s_state is not None:
            elapsed_s = duration_s_state
        else:
            if started_at_state:
                start_dt = _parse_datetime(started_at_state)
                elapsed_s = round((brasilia_now() - start_dt).total_seconds(), 3)
    except Exception:
        elapsed_s = None

    artifact_found = False
    artifact_path = None
    try:
        artifact_path = _resolve_artifact_path(None, symbol=symbol)
        artifact_found = True
    except Exception:
        artifact_found = False
        artifact_path = None

    if train_status in {"succeeded", "failed"} and not metrics_saved_state:
        try:
            save_train_metrics_snapshot(
                train_status=train_status,
                started_at=started_at_state,
                ended_at=ended_at_state,
                duration_s=duration_s_state,
                elapsed_s=elapsed_s,
                notebook=notebook_state,
                last_output_ipynb=last_output_ipynb_state,
                metrics=metrics_state or {},
                error=error_state,
            )
            with _TRAIN_LOCK:
                TRAIN_STATE["metrics_saved"] = True
                metrics_saved_state = True
        except Exception:
            pass

    return {
        "train_status": train_status,
        "phase": phase_state,
        "prereqs": prereqs_snapshot,
        "execute_engine": execute_engine_state,
        "execute_started_at": execute_started_at_state,
        "nbclient_started_at": nbclient_started_at_state,
        "started_at": started_at_state,
        "ended_at": ended_at_state,
        "duration_s": duration_s_state,
        "elapsed_s": elapsed_s,
        "error": error_state,
        "notebook": notebook_state,
        "last_output_ipynb": last_output_ipynb_state,
        "metrics": metrics_state,
        "artifact_found": artifact_found,
        "artifact_path": artifact_path,
    }