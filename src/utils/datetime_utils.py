"""Utilitários de data/hora focados no fuso de Brasília.

Contexto:
    Padroniza timestamps no fuso horário de Brasília (``America/Sao_Paulo``)
    com precisão de segundos. Em ambientes minimalistas (como algumas imagens
    Docker) a base de timezones do sistema pode não existir; nesse caso é
    aplicado um *fallback* para um offset fixo ``UTC-03:00``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def brasilia_now() -> datetime:
    """Retorna um ``datetime`` *timezone-aware* no fuso de Brasília.

    Contexto:
        Tenta utilizar ``zoneinfo.ZoneInfo("America/Sao_Paulo")`` quando
        disponível. Caso a base de timezones não esteja presente, faz
        *fallback* para um ``timezone`` fixo com offset de ``-3`` horas em
        relação ao UTC.

    Returns:
        datetime: Instante atual com informação de fuso horário.
    """
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/Sao_Paulo"))
    except Exception:
        return datetime.now(timezone(timedelta(hours=-3)))


def brasilia_iso(dt: datetime | None = None) -> str:
    """Converte um instante para string no fuso de Brasília.

    Contexto:
        Gera uma representação textual com precisão de segundos e offset
        explícito. Se o ``datetime`` de entrada não tiver ``tzinfo`` definido,
        ele é assumido como UTC antes da conversão para o fuso de Brasília.

    Args:
        dt: Instante opcional; se ``None``, utiliza o horário atual de Brasília.

    Returns:
        str: String no formato ``YYYY-MM-DD HH:MM:SS -03:00`` (ou equivalente
        com outro offset, se aplicável).
    """
    if dt is None:
        dt = brasilia_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    base = dt.isoformat(sep=" ", timespec="seconds")
    if len(base) >= 6 and base[-6] in {"+", "-"}:
        return f"{base[:-6]} {base[-6:]}"
    return base
