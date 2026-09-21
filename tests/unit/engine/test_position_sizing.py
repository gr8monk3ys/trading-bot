from unittest.mock import MagicMock

from brokers.protocol import Position
from engine.position_sizing import PositionSizer
from engine.session import PortfolioView


def _view(equity=100_000.0, buying_power=100_000.0, positions=()):
    return PortfolioView(
        equity=equity,
        cash=equity,
        positions={p.symbol: p for p in positions},
        buying_power=buying_power,
    )


def test_fixed_fraction_of_buying_power_then_capped_by_equity():
    s = PositionSizer(base_fraction=0.10, max_position_fraction=0.05).size("SPY", 100.0, _view())
    assert s.value == 5_000 and s.qty == 50 and s.fraction == 0.05
    assert "capped" in " ".join(s.steps)


def test_short_uses_the_short_fraction():
    s = PositionSizer(short_fraction=0.02, max_position_fraction=1.0).size(
        "SPY", 100.0, _view(), is_short=True
    )
    assert s.value == 2_000 and s.qty == 20


def test_kelly_when_enabled_and_short_haircut():
    kelly = MagicMock()
    kelly.calculate_position_size.return_value = (10_000.0, 0.10)
    sizer = PositionSizer(kelly=kelly, max_position_fraction=1.0)
    assert sizer.size("SPY", 100.0, _view()).qty == 100
    assert sizer.size("SPY", 100.0, _view(), is_short=True).qty == 80
    kelly.calculate_position_size.assert_called_with(current_capital=100_000.0, current_price=100.0)


def test_regime_multiplier_and_risk_haircut_apply_in_order():
    risk = MagicMock()
    risk.adjust_position_size.side_effect = lambda symbol, value, closes, held: value * 0.5
    sizer = PositionSizer(
        base_fraction=0.10,
        max_position_fraction=1.0,
        risk_manager=risk,
        regime_multiplier=lambda s: 0.5,
    )
    closes = [100.0] * 30
    held = (Position("QQQ", 10, 50.0, market_value=500.0),)
    s = sizer.size("SPY", 100.0, _view(positions=held), closes, held_closes={"QQQ": closes})
    assert s.value == 2_500  # 10% * 0.5 regime * 0.5 haircut
    assert risk.adjust_position_size.call_args.args[3]["QQQ"]["value"] == 500.0


def test_risk_manager_rejection_and_invalid_price_are_not_tradeable():
    risk = MagicMock()
    risk.adjust_position_size.return_value = 0.0
    assert not PositionSizer(risk_manager=risk).size("SPY", 100.0, _view(), [1.0] * 30).tradeable
    assert not PositionSizer().size("SPY", 0.0, _view()).tradeable
    assert (
        PositionSizer(risk_manager=risk).size("SPY", 100.0, _view(), [1.0] * 5).tradeable
    )  # too little history to haircut
