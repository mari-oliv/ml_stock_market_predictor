from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from sqlalchemy import Table, create_engine
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session, sessionmaker


class DataAccess:
    """Camada simples de acesso a dados via SQLAlchemy.

    Responsável por:
    - Criar o engine a partir da URL do banco
    - Fornecer sessões
    - Executar leituras e escritas básicas
    """

    def __init__(self, db_url: str) -> None:
        """Inicializa o acesso a dados com base na URL do banco.

        Args:
            db_url: URL de conexão do banco (ex.: "sqlite:///data/predictions.db").
        """
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Cria e retorna uma nova sessão SQLAlchemy."""
        return self.Session()

    def read_data(self, query: Any) -> Sequence[Row]:
        """Executa uma query de leitura e retorna todos os registros.

        Args:
            query: Objeto executável do SQLAlchemy (text/select/etc.).

        Returns:
            Sequência de linhas retornadas pela execução.
        """
        session = self.get_session()
        try:
            result = session.execute(query).fetchall()
            return result
        finally:
            session.close()

    def write_data(self, data: Any, table: Table) -> None:
        """Insere dados em uma tabela e efetiva a transação.

        Args:
            data: Dados a inserir (dict ou lista de dicts), compatível com `.values(...)`.
            table: Tabela SQLAlchemy (Table) com `.insert()`.
        """
        session = self.get_session()
        try:
            session.execute(table.insert().values(data))
            session.commit()
        finally:
            session.close()