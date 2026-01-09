from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import Table, create_engine
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session, sessionmaker


class DataAccess:
    """Camada simples de acesso a dados baseada em SQLAlchemy.

    Contexto:
        Centraliza a criação de ``engine`` e de sessões de banco,
        oferecendo operações básicas de leitura e escrita utilizadas
        pelos demais componentes da aplicação.
    """

    def __init__(self, db_url: str) -> None:
        """Inicializa o acesso a dados a partir da URL do banco.

        Contexto:
            Configura o ``engine`` SQLAlchemy e a fábrica de sessões
            (``sessionmaker``) usada pelas demais operações deste
            wrapper.

        Args:
            db_url: URL de conexão do banco (por exemplo,
                ``"sqlite:///data/predictions.db"``).

        Returns:
            None.
        """

        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Cria e retorna uma nova sessão SQLAlchemy.

        Contexto:
            Fornece uma sessão desacoplada, permitindo que chamadas
            externas controlem o ciclo de vida da transação quando
            necessário.

        Returns:
            Instância de :class:`sqlalchemy.orm.Session` pronta para
            uso em operações de banco.
        """

        return self.Session()

    def read_data(self, query: Any) -> Sequence[Row]:
        """Executa uma consulta de leitura e retorna todos os registros.

        Contexto:
            Abstrai a abertura/fechamento de sessão para operações de
            leitura simples, retornando o resultado completo em
            memória.

        Args:
            query: Objeto executável do SQLAlchemy (``text``,
                ``select`` ou similar).

        Returns:
            Sequência de linhas (:class:`sqlalchemy.engine.Row`)
            retornadas pela execução da consulta.
        """

        session = self.get_session()
        try:
            result = session.execute(query).fetchall()
            return result
        finally:
            session.close()

    def write_data(self, data: Any, table: Table) -> None:
        """Insere dados em uma tabela e confirma a transação.

        Contexto:
            Realiza uma operação de *insert* atômica, garantindo que a
            sessão seja sempre fechada, independentemente de sucesso ou
            erro.

        Args:
            data: Objeto compatível com ``.values(...)`` (``dict`` ou
                lista de ``dict``) representando as linhas a inserir.
            table: Instância de :class:`sqlalchemy.Table` na qual o
                *insert* será executado.

        Returns:
            None. Os efeitos são persistidos no banco via ``commit``.
        """

        session = self.get_session()
        try:
            session.execute(table.insert().values(data))
            session.commit()
        finally:
            session.close()