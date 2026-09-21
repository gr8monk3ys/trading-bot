"""Every strategy parameter, declared once with its default (ADR 0010).

A strategy's ``Params`` dataclass is the single source of its defaults;
``default_parameters()`` returns ``Params.defaults()`` and constructing a
strategy with a key that is not declared here raises. The four risk knobs
default from the settings module, so ``MAX_CORRELATION`` in ``.env`` reaches
the strategies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Mapping

from config import RISK_PARAMS


@dataclass
class BaseParams:
    allocation: Any = None
    daily_exits: bool = False
    drawdown_threshold: float = 0.3
    enable_short_selling: bool = False
    es_threshold: float = 0.04
    interval: int = 60
    max_correlation: float = field(default_factory=lambda: float(RISK_PARAMS["MAX_CORRELATION"]))
    max_daily_loss: float = 0.03
    max_position_size: float = 0.05
    max_positions: int = 5
    portfolio_risk_limit: float = field(
        default_factory=lambda: float(RISK_PARAMS["MAX_PORTFOLIO_RISK"])
    )
    position_risk_limit: float = field(
        default_factory=lambda: float(RISK_PARAMS["MAX_POSITION_RISK"])
    )
    position_size: float = 0.1
    price_history_window: int = 30
    sentiment_threshold: float = 0.6
    short_position_size: float = 0.08
    short_stop_loss: float = 0.04
    stop_loss: float = 0.03
    stop_loss_pct: float = 0.02
    symbols: Any = field(default_factory=lambda: [])
    take_profit: float = 0.05
    take_profit_pct: float = 0.05
    use_kelly_criterion: bool = False
    use_volatility_regime: bool = False
    var_confidence: float = field(default_factory=lambda: float(RISK_PARAMS["VAR_CONFIDENCE"]))
    var_threshold: float = 0.03
    volatility_threshold: float = 0.4

    kelly_fraction: float = 0.5
    kelly_lookback: int = 50
    kelly_min_trades: int = 100
    min_position_size: float = 0.01

    @classmethod
    def names(cls) -> set:
        return {f.name for f in fields(cls)}

    @classmethod
    def defaults(cls) -> Dict[str, Any]:
        return asdict(cls())

    @classmethod
    def check(cls, mapping: Mapping[str, Any]) -> Dict[str, Any]:
        """Reject keys this strategy does not declare; return defaults overlaid with mapping."""
        unknown = sorted(set(mapping) - cls.names())
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown parameter(s) {unknown}")
        merged = cls.defaults()
        merged.update(mapping)
        return merged


@dataclass
class MomentumParams(BaseParams):
    adx_period: int = 14
    adx_threshold: int = 25
    atr_multiplier: float = 2.0
    atr_period: int = 14
    bb_buy_threshold: float = 0.3
    bb_period: int = 20
    bb_sell_threshold: float = 0.7
    bb_std: float = 2.0
    crypto_long_only_buy_score_threshold: float = 1.0
    crypto_long_only_dip_buy_enabled: bool = True
    crypto_long_only_dip_min_macd_hist_delta: float = 0.02
    crypto_long_only_dip_min_rebound_pct: float = 0.001
    crypto_long_only_dip_rsi_max: float = 35.0
    crypto_long_only_relaxed_entry: bool = True
    fast_ma_period: int = 10
    macd_fast_period: int = 12
    macd_signal_period: int = 9
    macd_slow_period: int = 26
    max_portfolio_risk: float = 0.02
    max_position_risk: float = 0.01
    medium_ma_period: int = 20
    mtf_require_alignment: bool = True
    mtf_timeframes: Any = field(default_factory=lambda: ["5Min", "15Min", "1Hour"])
    position_size: float = 0.05
    position_size_pct: float = 0.1
    rsi_mode: str = "standard"
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_period: int = 14
    sizing_basis: str = "equity"
    slow_ma_period: int = 50
    trailing_activation_pct: float = 0.02
    trailing_stop_pct: float = 0.02
    use_bollinger_filter: bool = False
    use_multi_timeframe: bool = False
    use_trailing_stop: bool = True
    use_volatility_regime: bool = True
    volume_factor: float = 1.5
    volume_ma_period: int = 20


@dataclass
class MomentumBacktestParams(MomentumParams):
    """Daily-data overrides; every one explicit so nothing is inherited by accident."""

    adx_threshold: int = 20
    enable_short_selling: bool = True
    mtf_require_alignment: bool = False
    rsi_mode: str = "standard"
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_period: int = 14
    use_bollinger_filter: bool = True
    use_kelly_criterion: bool = False
    use_multi_timeframe: bool = False
    use_volatility_regime: bool = False
    volume_factor: float = 1.2


@dataclass
class MeanReversionParams(BaseParams):
    bb_period: int = 20
    bb_std: float = 2.0
    enable_short_selling: bool = True
    max_hold_days: int = 5
    max_portfolio_risk: float = 0.02
    max_position_risk: float = 0.01
    mean_lookback: int = 20
    mtf_require_alignment: bool = False
    mtf_timeframes: Any = field(default_factory=lambda: ["5Min", "15Min", "1Hour"])
    position_size_pct: float = 0.1
    profit_target_std: float = 0.5
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_period: int = 14
    short_stop_loss: float = 0.03
    sizing_basis: str = "equity"
    sma_period: int = 50
    std_threshold: float = 1.5
    stop_loss: float = 0.02
    take_profit: float = 0.04
    trailing_stop: float = 0.015
    use_multi_timeframe: bool = True


@dataclass
class AdaptiveParams(BaseParams):
    bear_strategy: str = "momentum_short"
    bull_strategy: str = "momentum"
    max_portfolio_risk: float = 0.02
    mean_reversion_take_profit: float = 0.04
    min_regime_confidence: float = 0.55
    regime_check_interval_minutes: int = 30
    sideways_strategy: str = "mean_reversion"
    use_kelly_criterion: bool = True
    use_multi_timeframe: bool = True
    use_trailing_stop: bool = True
    use_volatility_regime: bool = True


@dataclass
class SimpleMAParams(BaseParams):
    fast_period: int = 10
    slow_period: int = 30
