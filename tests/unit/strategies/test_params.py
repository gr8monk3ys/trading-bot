"""Every knob declared once (ADR 0010): unknown keys are errors, defaults are
present, and every declared parameter is read by something."""

import re
from pathlib import Path

import pytest

from strategies.adaptive_strategy import AdaptiveStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.momentum_strategy_backtest import MomentumStrategyBacktest
from strategies.params import (
    AdaptiveParams,
    BaseParams,
    MeanReversionParams,
    MomentumBacktestParams,
    MomentumParams,
    SimpleMAParams,
)
from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

REPO = Path(__file__).resolve().parents[3]
READERS = [*REPO.glob("strategies/*.py"), *REPO.glob("engine/*.py"), REPO / "main.py"]
READERS = [p for p in READERS if p.name != "params.py"]
SOURCE = "\n".join(p.read_text() for p in READERS)

# Declared for the process, consumed outside the strategies.
CONSUMED_ELSEWHERE = {
    "allocation": "StrategyManager bookkeeping",
    "max_daily_loss": "main.py wires it into the circuit breaker",
}


@pytest.mark.parametrize(
    "strategy_cls, params_cls",
    [
        (MomentumStrategy, MomentumParams),
        (MomentumStrategyBacktest, MomentumBacktestParams),
        (MeanReversionStrategy, MeanReversionParams),
        (AdaptiveStrategy, AdaptiveParams),
        (SimpleMACrossoverStrategy, SimpleMAParams),
    ],
)
def test_unknown_parameter_is_an_error_and_defaults_are_present(strategy_cls, params_cls):
    with pytest.raises(ValueError, match="unknown parameter"):
        strategy_cls(parameters={"posiiton_size": 0.1})
    strategy = strategy_cls(parameters={"symbols": ["SPY"]})
    assert set(strategy.parameters) == params_cls.names()
    assert strategy.parameters["symbols"] == ["SPY"]


def test_backtest_variant_overrides_only_the_daily_knobs():
    assert MomentumBacktestParams.defaults()["rsi_mode"] == "standard"
    assert MomentumBacktestParams.defaults()["use_kelly_criterion"] is False
    assert MomentumParams.names() == MomentumBacktestParams.names()


def test_risk_knobs_default_from_settings():
    from config import RISK_PARAMS

    assert BaseParams.defaults()["max_correlation"] == RISK_PARAMS["MAX_CORRELATION"]
    assert BaseParams.defaults()["var_confidence"] == RISK_PARAMS["VAR_CONFIDENCE"]


@pytest.mark.parametrize(
    "params_cls", [MomentumParams, MeanReversionParams, AdaptiveParams, SimpleMAParams]
)
def test_every_declared_parameter_is_read_somewhere(params_cls):
    unread = [
        name
        for name in sorted(params_cls.names())
        if name not in CONSUMED_ELSEWHERE and not re.search(rf'["\']{name}["\']', SOURCE)
    ]
    assert unread == [], f"{params_cls.__name__} declares parameters nothing reads: {unread}"
