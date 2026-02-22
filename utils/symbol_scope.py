"""
Utilities for symbol-scope filtering across multi-process live sessions.

When multiple bot processes share the same broker account, each process should
only ingest positions for its own configured symbol universe.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

from utils.crypto_utils import is_crypto_symbol, normalize_crypto_symbol


def canonical_symbol(symbol: str) -> str:
    """
    Canonicalize symbol strings for scope comparisons.

    Crypto symbols are normalized to compact `BASEQUOTE` form (e.g. BTCUSD)
    so both `BTC/USD` and `BTCUSD` match the same scope key.
    """
    normalized = (symbol or "").upper().strip().replace("-", "").replace("_", "")
    if not normalized:
        return ""

    if is_crypto_symbol(normalized):
        try:
            return normalize_crypto_symbol(normalized).replace("/", "")
        except ValueError:
            return normalized.replace("/", "")

    return normalized


def build_symbol_scope(symbols: Optional[Iterable[str]]) -> Optional[Set[str]]:
    """Build a canonical symbol set; `None` means no filtering."""
    if symbols is None:
        return None

    scope: Set[str] = set()
    for symbol in symbols:
        canonical = canonical_symbol(str(symbol))
        if canonical:
            scope.add(canonical)
    return scope


def symbol_in_scope(symbol: str, scope: Optional[Set[str]]) -> bool:
    """Return True when the symbol belongs to the provided scope."""
    if scope is None:
        return True
    return canonical_symbol(symbol) in scope
