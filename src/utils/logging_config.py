"""Configuração central de logging da aplicação.

Define formato e nível de logs via variável de ambiente `LOG_LEVEL` e garante
que logs de `uvicorn`/`fastapi` sigam o mesmo nível.
"""

import logging
import os
import sys


def setup_logging() -> logging.Logger:
    """Configura o logging global e retorna o logger principal da aplicação.

    Lê `LOG_LEVEL` do ambiente, configura `logging.basicConfig` com handler para
    stdout e ajusta o nível de loggers relacionados ao Uvicorn/FastAPI.

    Returns:
        Logger principal com nome "app".
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logging.getLogger(name).setLevel(level)

    logger = logging.getLogger("app")
    logger.setLevel(level)
    return logger