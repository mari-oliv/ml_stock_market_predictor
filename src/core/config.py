"""Configuração central do projeto baseada em variáveis de ambiente.

Contexto:
    Responsável por carregar valores do arquivo ``.env`` (quando
    existente) e expor parâmetros globais de configuração para o
    restante da aplicação, como caminhos de modelo, URL de banco
    de dados e versão da API.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Namespace estático com configurações carregadas do ambiente.

    Contexto:
        Lido pelos diferentes componentes da aplicação (API, treino
        e inferência) para obter valores de configuração padronizados.

    Atributos de classe:
        DEBUG: Flag booleana indicando execução em modo debug.
        DATABASE_URL: URL de conexão com o banco de dados.
        MODEL_PATH: Caminho base onde modelos treinados são armazenados.
        FEATURE_SET: Nome do conjunto de features a ser utilizado.
        API_VERSION: Versão exposta nos endpoints da API.
    """

    DEBUG = os.getenv("DEBUG", "False") == "True"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///default.db")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/")
    FEATURE_SET = os.getenv("FEATURE_SET", "default_features")
    API_VERSION = os.getenv("API_VERSION", "v1")