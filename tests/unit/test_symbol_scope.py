from utils.symbol_scope import build_symbol_scope, canonical_symbol, symbol_in_scope


def test_canonical_symbol_normalizes_crypto_variants() -> None:
    assert canonical_symbol("BTC/USD") == "BTCUSD"
    assert canonical_symbol("btcusd") == "BTCUSD"
    assert canonical_symbol("btc-usd") == "BTCUSD"


def test_build_symbol_scope_drops_empty_values() -> None:
    scope = build_symbol_scope(["AAPL", "BTC/USD", "", "  "])
    assert scope == {"AAPL", "BTCUSD"}


def test_symbol_in_scope_allows_all_when_scope_is_none() -> None:
    assert symbol_in_scope("AAPL", None) is True
    assert symbol_in_scope("BTCUSD", None) is True


def test_symbol_in_scope_uses_canonical_matching() -> None:
    scope = build_symbol_scope(["BTC/USD", "MSFT"])
    assert symbol_in_scope("BTCUSD", scope) is True
    assert symbol_in_scope("msft", scope) is True
    assert symbol_in_scope("ETH/USD", scope) is False
