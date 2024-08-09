"""
RiskManager sub-package.

The original ``strategies/risk_manager.py`` (~1062 LOC) mixed risk
calculations (vol, all VaR methods, ES, max drawdown, position risk,
correlation, portfolio risk) with risk enforcement (position sizing,
limit enforcement, margin monitoring, halt decisions). Split via the
same mixin pattern as phases 5-7 into focused modules:

    - calculator.py — pure risk math (``RiskCalculatorMixin``) plus the
                      ``safe_divide`` helper and the ``RiskCalculationError``
                      exception class
    - enforcer.py   — position sizing + limits + margin + halt logic
                      (``RiskEnforcerMixin``)

This ``__init__.py`` is the ~50 LOC facade that wires the two mixins
together into the public ``RiskManager`` class and re-exports the helper
symbols. External callers continue to do
``from strategies.risk_manager import RiskManager`` (and
``RiskCalculationError`` / ``safe_divide``) exactly as before.
"""

from strategies.risk_manager.calculator import (
    RiskCalculationError,
    RiskCalculatorMixin,
    safe_divide,
)
from strategies.risk_manager.enforcer import RiskEnforcerMixin


class RiskManager(RiskCalculatorMixin, RiskEnforcerMixin):
    """
    Portfolio risk manager combining risk calculations and enforcement.

    Public API and behavior identical to the pre-refactor single-file
    ``RiskManager`` — see ``calculator.py`` and ``enforcer.py`` for the
    underlying implementations.
    """

    def __init__(
        self,
        max_portfolio_risk=0.02,
        max_position_risk=0.01,
        max_correlation=0.7,
        volatility_threshold=0.4,
        var_threshold=0.03,
        es_threshold=0.04,
        drawdown_threshold=0.3,
        strict_correlation_enforcement=True,  # NEW: Reject instead of just adjusting
    ):
        # P2 FIX: Validate all thresholds to prevent invalid risk calculations
        self._validate_threshold("max_portfolio_risk", max_portfolio_risk, 0, 1)
        self._validate_threshold("max_position_risk", max_position_risk, 0, 1)
        self._validate_threshold("max_correlation", max_correlation, -1, 1)
        self._validate_threshold("volatility_threshold", volatility_threshold, 0.001, 10)
        self._validate_threshold("var_threshold", var_threshold, 0.001, 1)
        self._validate_threshold("es_threshold", es_threshold, 0.001, 1)
        self._validate_threshold("drawdown_threshold", drawdown_threshold, 0.001, 1)

        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.volatility_threshold = volatility_threshold
        self.var_threshold = var_threshold
        self.es_threshold = es_threshold
        self.drawdown_threshold = drawdown_threshold
        self.strict_correlation_enforcement = strict_correlation_enforcement
        self.position_sizes = {}
        self.position_correlations = {}

    @staticmethod
    def _validate_threshold(name: str, value: float, min_val: float, max_val: float):
        """P2 FIX: Validate that threshold values are within acceptable bounds."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value)}")
        if value < min_val or value > max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
        if value == 0 and name.endswith("_threshold"):
            raise ValueError(f"{name} cannot be zero (would cause division by zero)")


__all__ = [
    "RiskManager",
    "RiskCalculationError",
    "RiskCalculatorMixin",
    "RiskEnforcerMixin",
    "safe_divide",
]
