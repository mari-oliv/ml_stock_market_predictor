"""Persistência de predições e métricas em banco de dados via SQLAlchemy.

Contexto:
    Este módulo resolve a URL do banco a partir da variável de ambiente
    ``DATABASE_URL`` (quando definida) e, caso contrário, utiliza um SQLite
    local em ``data/predictions.db``. Ele também garante que o schema mínimo
    exista antes de inserir registros tanto na tabela de predições quanto na
    de métricas de treino.
"""

import os

from sqlalchemy import create_engine, text

from src.utils.datetime_utils import brasilia_iso


def _project_root() -> str:
    """Retorna o caminho absoluto da raiz do projeto.

    Contexto:
        A raiz do projeto é inferida a partir da localização deste módulo,
        caminhando dois níveis acima em relação ao diretório atual.

    Returns:
        str: Caminho absoluto da raiz do projeto.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def _default_sqlite_url() -> str:
    """Monta a URL do SQLite local e garante a existência do diretório ``data/``.

    Contexto:
        Usado como fallback quando ``DATABASE_URL`` não está configurada. O
        arquivo SQLite é criado em ``data/predictions.db`` dentro da raiz do
        projeto, criando o diretório ``data/`` caso ainda não exista.

    Returns:
        str: URL de conexão no formato ``sqlite:///.../data/predictions.db``.
    """
    root = _project_root()
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return f"sqlite:///{os.path.join(data_dir, 'predictions.db')}"


DB_URL = os.getenv("DATABASE_URL", _default_sqlite_url())
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)


def _ensure_schema() -> None:
    """Cria a tabela ``predictions`` se não existir.

    Contexto:
        Executa um DDL padrão (com tipos "grandes") e, em caso de falha
        (por exemplo, em SQLite), aplica um DDL alternativo compatível com
        SQLite. Deve ser chamado antes de qualquer inserção em ``predictions``.

    Returns:
        None
    """
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
    """Cria a tabela ``metrics`` se não existir.

    Contexto:
        A tabela armazena snapshots de métricas do treino (uma linha por
        execução finalizada), incluindo tempos de cada etapa, engine utilizada
        e informações do kernel do notebook.

    Returns:
        None
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
    """Salva uma predição na tabela ``predictions``.

    Contexto:
        Insere uma nova linha na tabela de predições, registrando o símbolo,
        o valor previsto, a data de referência e o timestamp de criação em
        horário de Brasília.

    Args:
        symbol: Ticker/símbolo associado à predição.
        value: Valor previsto.
        date_iso: Data/hora (ISO-8601) referente ao valor previsto.

    Returns:
        None
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
    """Salva um snapshot das métricas do treino na tabela ``metrics``.

    Contexto:
        Esta função é pensada para ser chamada ao final de uma execução de
        treino (bem-sucedida ou não), consolidando em uma única linha as
        principais métricas e metadados da execução.

    Args:
        train_status: Status final do treino (por exemplo, ``"succeeded"`` ou
            ``"failed"``).
        started_at: Momento de início do treino em formato ISO-8601, ou ``None``.
        ended_at: Momento de término do treino em formato ISO-8601, ou ``None``.
        duration_s: Duração total (em segundos) ou ``None``.
        elapsed_s: Tempo decorrido reportado (em segundos) ou ``None``.
        notebook: Caminho do notebook executado, ou ``None``.
        last_output_ipynb: Caminho do notebook de saída (se existir), ou ``None``.
        metrics: Dicionário com métricas e metadados adicionais (engine, tempos
            por etapa, kernel, etc.).
        error: Mensagem de erro associada à execução, ou ``None``.

    Returns:
        None
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