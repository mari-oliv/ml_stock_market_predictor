"""Configuração central do projeto via variáveis de ambiente.

Carrega valores do arquivo `.env` (se existir) e expõe configurações em uma
classe estática para uso no restante da aplicação.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configurações da aplicação carregadas do ambiente."""

    DEBUG = os.getenv("DEBUG", "False") == "True"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///default.db")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/")
    FEATURE_SET = os.getenv("FEATURE_SET", "default_features")
    API_VERSION = os.getenv("API_VERSION", "v1")