"""Persistência de predições em banco de dados via SQLAlchemy.

Este módulo resolve a URL do banco a partir de `DATABASE_URL` (quando definido)
e, caso contrário, utiliza um SQLite local em `data/predictions.db`. Também
garante que o schema mínimo exista antes de inserir registros.
"""

import os

from sqlalchemy import create_engine, text

from src.utils.datetime_utils import brasilia_iso


def _project_root() -> str:
    """Retorna o caminho absoluto da raiz do projeto a partir deste módulo."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def _default_sqlite_url() -> str:
    """Monta a URL do SQLite local e garante a existência do diretório `data/`."""
    root = _project_root()
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return f"sqlite:///{os.path.join(data_dir, 'predictions.db')}"


DB_URL = os.getenv("DATABASE_URL", _default_sqlite_url())
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)


def _ensure_schema() -> None:
    """Cria a tabela `predictions` se não existir, com fallback para SQLite."""
    ddl = """
    CREATE TABLE IF NOT EXISTS predictions (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        symbol VARCHAR(64) NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        date TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    sqlite_ddl = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        value REAL NOT NULL,
        date TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    with engine.begin() as conn:
        try:
            conn.execute(text(ddl))
        except Exception:
            conn.execute(text(sqlite_ddl))


def _ensure_metrics_schema() -> None:
    """Cria a tabela `metrics` se não existir.

    A tabela armazena snapshots de métricas do treino (uma linha por execução finalizada).
    """
    sqlite_ddl = """
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        train_status TEXT NOT NULL,
        started_at TEXT,
        ended_at TEXT,
        duration_s REAL,
        elapsed_s REAL,
        notebook TEXT,
        last_output_ipynb TEXT,
        engine TEXT,
        find_notebook_s REAL,
        prepare_paths_s REAL,
        execute_s REAL,
        kernel_name TEXT,
        kernel_startup_timeout_s INTEGER,
        error TEXT,
        created_at TEXT NOT NULL
    )
    """
    with engine.begin() as conn:
        conn.execute(text(sqlite_ddl))


_ensure_schema()
_ensure_metrics_schema()


def save_prediction(symbol: str, value: float, date_iso: str) -> None:
    """Salva uma predição na tabela `predictions`.

    Args:
        symbol: Ticker/símbolo associado à predição.
        value: Valor previsto.
        date_iso: Data/hora (ISO-8601) referente ao valor previsto.
    """
    created_iso = brasilia_iso()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions (symbol, value, date, created_at) "
                "VALUES (:t, :v, :d, :c)"
            ),
            {"t": symbol, "v": float(value), "d": date_iso, "c": created_iso},
        )


def save_train_metrics_snapshot(
    *,
    train_status: str,
    started_at: str | None,
    ended_at: str | None,
    duration_s: float | None,
    elapsed_s: float | None,
    notebook: str | None,
    last_output_ipynb: str | None,
    metrics: dict,
    error: str | None,
) -> None:
    """Salva um snapshot das métricas do treino na tabela `metrics`.

    Args:
        train_status: Status final do treino (ex.: "succeeded" ou "failed").
        started_at: ISO-8601 do início.
        ended_at: ISO-8601 do fim.
        duration_s: Duração total (s).
        elapsed_s: Tempo decorrido reportado (s).
        notebook: Caminho do notebook executado.
        last_output_ipynb: Caminho do notebook de saída (se existir).
        metrics: Dict de métricas (engine, tempos por etapa, kernel, etc.).
        error: Mensagem de erro (se houver).
    """
    created_at = brasilia_iso()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO metrics (
                    train_status,
                    started_at,
                    ended_at,
                    duration_s,
                    elapsed_s,
                    notebook,
                    last_output_ipynb,
                    engine,
                    find_notebook_s,
                    prepare_paths_s,
                    execute_s,
                    kernel_name,
                    kernel_startup_timeout_s,
                    error,
                    created_at
                ) VALUES (
                    :train_status,
                    :started_at,
                    :ended_at,
                    :duration_s,
                    :elapsed_s,
                    :notebook,
                    :last_output_ipynb,
                    :engine,
                    :find_notebook_s,
                    :prepare_paths_s,
                    :execute_s,
                    :kernel_name,
                    :kernel_startup_timeout_s,
                    :error,
                    :created_at
                )
                """
            ),
            {
                "train_status": train_status,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_s": duration_s,
                "elapsed_s": elapsed_s,
                "notebook": notebook,
                "last_output_ipynb": last_output_ipynb,
                "engine": metrics.get("engine"),
                "find_notebook_s": metrics.get("find_notebook_s"),
                "prepare_paths_s": metrics.get("prepare_paths_s"),
                "execute_s": metrics.get("execute_s"),
                "kernel_name": metrics.get("kernel_name"),
                "kernel_startup_timeout_s": metrics.get("kernel_startup_timeout_s"),
                "error": error,
                "created_at": created_at,
            },
        )