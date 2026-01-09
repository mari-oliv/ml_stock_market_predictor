import contextlib
import logging
import time


def _fmt_extra(extra: dict | None) -> str:
    """Formata um dicionário `extra` como sufixo `k=v` para logs.

    Args:
        extra: Dicionário opcional com campos adicionais a serem logados.

    Returns:
        String vazia quando `extra` é None/vazio; caso contrário, retorna uma string
        iniciando com espaço e pares `k=v` separados por espaço.
    """
    if not extra:
        return ""
    return " " + " ".join(f"{k}={v}" for k, v in extra.items())


@contextlib.contextmanager
def section(name: str, logger: logging.Logger, extra: dict | None = None):
    """Context manager para medir duração e logar início/fim de uma seção.

    Registra um log `{name}:start` ao entrar e `{name}:end` ao sair, incluindo
    status ("ok" ou "error") e `duration_ms`.

    Args:
        name: Nome da seção (prefixo das mensagens de log).
        logger: Logger a ser utilizado.
        extra: Dicionário opcional de campos adicionais para anexar ao log.

    Yields:
        None.

    Raises:
        Repropaga qualquer exceção levantada dentro do bloco, marcando status como "error".
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