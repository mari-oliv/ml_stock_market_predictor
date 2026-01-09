"""Utilitários de data/hora.

Padroniza timestamps no fuso de Brasília (America/Sao_Paulo) com precisão de segundos.

Observação: Em imagens Docker minimalistas, a base de timezones do sistema pode não existir.
Neste caso, fazemos fallback para um offset fixo UTC-03:00.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def brasilia_now() -> datetime:
    """Retorna um datetime timezone-aware no fuso de Brasília."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/Sao_Paulo"))
    except Exception:
        return datetime.now(timezone(timedelta(hours=-3)))


def brasilia_iso(dt: datetime | None = None) -> str:
    """Converte para string no fuso de Brasília, com segundos e offset.

    Args:
        dt: datetime opcional; se None, usa o horário atual de Brasília.

    Returns:
        String no formato `YYYY-MM-DD HH:MM:SS -03:00`.
    """
    if dt is None:
        dt = brasilia_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    base = dt.isoformat(sep=" ", timespec="seconds")
    if len(base) >= 6 and base[-6] in {"+", "-"}:
        return f"{base[:-6]} {base[-6:]}"
    return base
