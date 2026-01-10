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
    """Modelo de entrada para o endpoint de predição ``POST /predict``.

    Contexto:
        Representa o payload recebido pela API para geração de uma
        predição de preço de ativo, permitindo escolher a origem dos
        dados (yfinance ou CSV local) e um artefato de modelo opcional.

    Atributos:
        symbol: Código do ativo a ser previsto (por exemplo, "PETR4.SA").
        data_source: Origem dos dados a ser utilizada ("auto", "yfinance" ou "csv").
        csv_path: Caminho opcional para um CSV local contendo a série histórica.
        csv_target_col: Nome opcional da coluna alvo dentro do CSV informado.
        artifact_path: Caminho opcional para o artefato de modelo já treinado.
    """

    symbol: str
    data_source: Literal["auto", "yfinance", "csv"] = "auto"
    csv_path: Optional[str] = None
    csv_target_col: Optional[str] = None
    artifact_path: Optional[str] = None


@router.get("/")
def health():
    """Verifica a saúde básica da API.

    Contexto:
        Usado por ferramentas de monitoramento ou orquestração para
        confirmar que o serviço HTTP está respondendo.

    Returns:
        Dicionário simples com a chave ``status`` indicando operação normal.
    """

    return {"status": "api is working"}


@router.post("/predict")
def predict(req: PredictRequest):
    """Gera a próxima predição de preço e persiste o resultado.

    Contexto:
        Orquestra a inferência do próximo valor de preço usando o
        artefato de modelo disponível e a fonte de dados configurada.
        Quando possível, registra a predição em armazenamento interno
        para fins de histórico.

    Args:
        req: Instância de :class:`PredictRequest` contendo símbolo,
            origem dos dados e caminhos opcionais de CSV/artefato.

    Returns:
        Dicionário com valor previsto (arredondado), símbolo, data de
        referência e metadados sobre a fonte de dados utilizada.

    Raises:
        HTTPException: Em caso de falha na etapa de predição ou
            carregamento de dados/modelo, com código 500.
    """
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
    """Converte uma string de timestamp em instância ``datetime``.

    Contexto:
        Normaliza diferentes variações de formatação de timezone
        (com ou sem espaço antes do offset) para um único objeto
        ``datetime`` utilizável em cálculos de duração.

    Args:
        value: String com o timestamp em formato ISO compatível,
            por exemplo ``"YYYY-MM-DDTHH:MM:SS-03:00"`` ou variantes
            com espaços.

    Returns:
        Objeto :class:`datetime.datetime` representando o instante
        informado na string.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        parts = value.split(" ")
        if len(parts) == 3:
            return datetime.fromisoformat(f"{parts[0]} {parts[1]}{parts[2]}")
        raise


def _project_root() -> str:
    """Obtém o caminho absoluto da raiz do projeto.

    Contexto:
        Parte da localização deste módulo para subir na hierarquia
        de diretórios até a raiz do repositório, permitindo resolver
        caminhos relativos de notebooks e artefatos.

    Returns:
        Caminho absoluto da raiz do projeto em forma de string.
    """

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def _collect_train_prereqs() -> dict[str, Any]:
    """Coleta informações sobre kernel Jupyter e dependências de treino.

    Contexto:
        Verifica se bibliotecas necessárias ao runner de notebooks
        estão instaladas e se o kernel configurado está disponível,
        permitindo validar o ambiente antes de disparar o treino.

    Returns:
        Dicionário com timestamp da checagem, nome do kernel,
        disponibilidade do kernel, lista de módulos ausentes e
        flag booleana ``ok`` indicando se o ambiente está pronto.
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
    """Localiza o notebook de treino a partir da raiz do projeto.

    Contexto:
        Busca o caminho do ``notebook.ipynb`` usando variáveis de
        ambiente de override e, em seguida, padrões conhecidos de
        diretórios, com fallback para busca recursiva.

    Args:
        root: Caminho absoluto considerado como raiz do projeto.

    Returns:
        Caminho absoluto para o notebook de treino encontrado.

    Raises:
        FileNotFoundError: Se nenhum notebook de treino for localizado.
    """
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
    """Executa o notebook de treino e atualiza o estado global.

    Contexto:
        Função interna disparada em thread dedicada para executar o
        notebook de treinamento, medir tempos das etapas principais e
        registrar métricas/erros em ``TRAIN_STATE``.

    Returns:
        None. Os efeitos são observados via mutação de ``TRAIN_STATE``
        e logs, além de eventual gravação de métricas persistentes.
    """
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
    """Dispara o processo de treinamento em background.

    Contexto:
        Endpoint responsável apenas por iniciar o fluxo de treinamento
        de modelo em uma thread separada. Opcionalmente, permite
        sobrescrever o notebook padrão via parâmetro, realizando uma
        validação rápida de existência do arquivo antes de iniciar.
        O acompanhamento detalhado do progresso deve ser feito via
        endpoint ``GET /check_train``.

    Args:
        notebook: Caminho opcional para um notebook específico de treino.
            Quando informado e encontrado, sobrescreve o caminho padrão
            por meio da variável de ambiente ``TRAIN_NOTEBOOK``.

    Returns:
        Dicionário contendo:
            - ``status``: "started" ou "running" se já houver treino em andamento;
            - ``started_at``: timestamp do momento em que o treino foi disparado;
            - ``notebook``: caminho do notebook de treino resolvido;
            - ``message``: instrução breve sobre aguardar e consultar o status.

    Raises:
        HTTPException:
            - 400 se o notebook informado não existir;
            - 400 se não for possível resolver um notebook de treino padrão.
    """

    if TRAIN_STATE["status"] == "running":
        logger.info("train:already_running")
        return {"status": "running", "started_at": TRAIN_STATE["started_at"]}

    if notebook:
        nb_path = os.path.abspath(os.path.expanduser(notebook))
        if not os.path.exists(nb_path):
            logger.warning(f"train:override_notebook not_found path={nb_path}")
        else:
            os.environ["TRAIN_NOTEBOOK"] = nb_path
            with _TRAIN_LOCK:
                TRAIN_STATE["notebook"] = nb_path
            logger.info(f"train:override_notebook path={nb_path}")


    t = threading.Thread(target=_run_notebook, daemon=True)
    t.start()
    now = brasilia_iso()
    logger.info("train:started")

    return {
        "status": "started",
        "started_at": now,
        "notebook": TRAIN_STATE.get("notebook"),
        "message": "Treino do modelo iniciado em background; consulte /check_train para acompanhar o status.",
    }


@router.get("/check_train")
def check_train(symbol: Optional[str] = None):
    """Consulta o status do processo de treinamento e disponibilidade de artefato.

    Contexto:
        Permite acompanhar o andamento do treino disparado em
        background, além de informar se já existe artefato salvo
        para um símbolo específico quando fornecido.

    Args:
        symbol: Código opcional do ativo para checagem de existência
            de artefato treinado associado.

    Returns:
        Dicionário contendo estado atual do treino, métricas
        temporais, informações de pré-requisitos do ambiente,
        status de salvamento de métricas e presença/caminho do
        artefato para o símbolo consultado.
    """
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