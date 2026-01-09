"""Utilitários simples para medição de tempo e logging de seções.

Contexto:
    Fornece um *context manager* que registra logs de início/fim de uma
    "seção" de código, incluindo a duração em milissegundos e um status
    indicando sucesso ou erro.
"""

import contextlib
import logging
import time


def _fmt_extra(extra: dict | None) -> str:
    """Formata um dicionário ``extra`` como sufixo ``k=v`` para logs.

    Contexto:
        Usado internamente para serializar campos adicionais de contexto em
        logs de seções temporizadas.

    Args:
        extra: Dicionário opcional com campos adicionais a serem logados.

    Returns:
        str: String vazia quando ``extra`` é ``None``/vazio; caso contrário,
        uma string iniciando com espaço e pares ``k=v`` separados por espaço.
    """
    if not extra:
        return ""
    return " " + " ".join(f"{k}={v}" for k, v in extra.items())


@contextlib.contextmanager
def section(name: str, logger: logging.Logger, extra: dict | None = None):
    """Context manager para medir duração e logar início/fim de uma seção.

    Contexto:
        Registra um log ``"{name}:start"`` ao entrar no bloco e
        ``"{name}:end"`` ao sair, incluindo um ``status`` (``"ok"`` ou
        ``"error"``) e o campo ``duration_ms`` com a duração em
        milissegundos, além de eventuais campos extras formatados por
        :func:`_fmt_extra`.

    Args:
        name: Nome da seção (prefixo das mensagens de log).
        logger: Logger a ser utilizado para registrar as mensagens.
        extra: Dicionário opcional de campos adicionais para anexar ao log.

    Yields:
        None: Permite envolver um bloco de código cujo tempo de execução será medido.

    Raises:
        Repropaga qualquer exceção levantada dentro do bloco, marcando
        ``status="error"`` no log de saída.
    """
    start = time.perf_counter()
    logger.info(f"{name}:start{_fmt_extra(extra)}")
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        status = "ok" if ok else "error"
        logger.info(f"{name}:end status={status} duration_ms={elapsed_ms:.2f}{_fmt_extra(extra)}")