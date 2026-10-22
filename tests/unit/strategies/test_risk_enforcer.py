"""
Tests for strategies/risk_manager/enforcer.py (RiskEnforcerMixin).

Exercised through the real ``RiskManager`` facade so the calculator mixin's
math (VaR, ES, volatility, drawdown, correlation, portfolio risk) feeds the
enforcement decisions exactly as it does in production. Nothing is mocked.

Covers:
- adjust_position_size: risk / portfolio down-scaling math and data-error fail-safe
- enforce_limits: input validation, each violation type, pass-through, error path
- get_risk_summary: pass/fail flags against thresholds, invalid input
- calculate_margin_requirement / check_margin_status: status bands, boundaries,
  deficit and reduction arithmetic
- get_liquidation_prices / should_halt_for_margin
"""

import numpy as np
import pytest

from strategies.risk_manager import RiskManager

DESIRED_SIZE = 1000.0
RANDOM_SEED = 7


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def risk_manager():
    """RiskManager with production defaults."""
    return RiskManager()


@pytest.fixture
def lenient_manager():
    """RiskManager whose per-symbol limits are at their maxima.

    Only the correlation limit (0.7) is left meaningful, so a test can isolate
    a single violation type.
    """
    return RiskManager(
        max_portfolio_risk=1.0,
        max_position_risk=1.0,
        var_threshold=1.0,
        es_threshold=1.0,
        volatility_threshold=10,
        drawdown_threshold=1.0,
    )


@pytest.fixture
def flat_prices():
    """Constant prices: zero volatility, VaR, ES and drawdown."""
    return [100.0] * 30


@pytest.fixture
def noisy_prices():
    """Mild random walk around 100."""
    rng = np.random.default_rng(RANDOM_SEED)
    return (100 + np.cumsum(rng.normal(0, 0.3, 60))).tolist()


@pytest.fixture
def crash_prices():
    """Steep sawtooth decline: ~86% peak-to-trough drawdown."""
    return [100 * (0.9**i) * (1.02 if i % 2 else 1.0) for i in range(20)]


# =============================================================================
# adjust_position_size (the path engine/position_sizing.py uses)
# =============================================================================
class TestAdjustPositionSizeScaling:
    def test_scales_down_by_position_risk_ratio(self, risk_manager, noisy_prices):
        risk = risk_manager.calculate_position_risk("X", noisy_prices)
        assert risk > risk_manager.max_position_risk

        adjusted = risk_manager.adjust_position_size("X", DESIRED_SIZE, noisy_prices, {})

        expected = DESIRED_SIZE * risk_manager.max_position_risk / risk
        assert adjusted == pytest.approx(expected)
        assert 0 < adjusted < DESIRED_SIZE

    def test_portfolio_risk_limit_binds_when_it_is_tighter(self, noisy_prices):
        rm = RiskManager(max_position_risk=1.0, max_portfolio_risk=0.02)
        positions = {"X": {"value": 5000.0}}

        adjusted = rm.adjust_position_size("X", DESIRED_SIZE, noisy_prices, positions)

        # A single held position has weight 1, so portfolio risk == its risk score.
        risk = positions["X"]["risk"]
        assert adjusted == pytest.approx(DESIRED_SIZE * 0.02 / risk)

    def test_zero_risk_keeps_full_size(self, risk_manager, flat_prices):
        assert risk_manager.adjust_position_size("X", DESIRED_SIZE, flat_prices, {}) == (
            DESIRED_SIZE
        )

    def test_malformed_position_entry_fails_safe_to_zero(self, risk_manager, noisy_prices):
        # The held entry is not a dict; writing its risk raises TypeError.
        assert risk_manager.adjust_position_size("X", DESIRED_SIZE, noisy_prices, {"X": None}) == 0


# =============================================================================
# enforce_limits
# =============================================================================
class TestEnforceLimitsValidation:
    @pytest.mark.parametrize("desired_size", [0, -1, -1000.0])
    def test_non_positive_size_rejected(self, risk_manager, noisy_prices, desired_size):
        size, violations = risk_manager.enforce_limits("X", desired_size, noisy_prices, {})
        assert size == 0
        assert list(violations) == ["invalid_size"]

    @pytest.mark.parametrize(
        "history, points",
        [(None, 0), ([], 0), ([100.0], 1)],
    )
    def test_insufficient_history_rejected(self, risk_manager, history, points):
        size, violations = risk_manager.enforce_limits("X", DESIRED_SIZE, history, {})
        assert size == 0
        assert list(violations) == ["insufficient_data"]
        assert f"{points} points" in violations["insufficient_data"]

    def test_non_numeric_history_reports_calculation_error(self, risk_manager):
        size, violations = risk_manager.enforce_limits("X", DESIRED_SIZE, ["a", "b"], {})
        assert size == 0
        assert list(violations) == ["calculation_error"]


class TestEnforceLimitsDecisions:
    def test_flat_prices_pass_with_full_size(self, risk_manager, flat_prices):
        size, violations = risk_manager.enforce_limits("X", DESIRED_SIZE, flat_prices, {})
        assert size == DESIRED_SIZE
        assert violations == {}

    def test_crash_series_rejected_on_every_tail_metric(self, risk_manager, crash_prices):
        size, violations = risk_manager.enforce_limits("X", DESIRED_SIZE, crash_prices, {})
        assert size == 0
        assert {
            "var_exceeded",
            "es_exceeded",
            "drawdown_exceeded",
            "portfolio_risk_exceeded",
        } <= set(violations)

    def test_volatility_limit(self):
        rm = RiskManager(
            max_portfolio_risk=1.0,
            var_threshold=1.0,
            es_threshold=1.0,
            drawdown_threshold=1.0,
            volatility_threshold=0.1,
        )
        # Alternating +/-1% returns: ~16% annualised vol, small drawdown.
        prices = [100.0, 101.0] * 15
        size, violations = rm.enforce_limits("X", DESIRED_SIZE, prices, {})
        assert size == 0
        assert list(violations) == ["volatility_exceeded"]

    def test_portfolio_risk_limit_alone(self, noisy_prices):
        rm = RiskManager(
            max_portfolio_risk=0.02,
            var_threshold=1.0,
            es_threshold=1.0,
            volatility_threshold=10,
            drawdown_threshold=1.0,
        )
        size, violations = rm.enforce_limits("X", DESIRED_SIZE, noisy_prices, {})
        assert size == 0
        assert list(violations) == ["portfolio_risk_exceeded"]

    def test_correlated_holding_rejected_and_named(self, lenient_manager, noisy_prices):
        positions = {
            "Y": {"value": 500.0, "price_history": list(noisy_prices)},
            "Z": {"value": 500.0},  # no price_history: skipped for correlation
        }
        size, violations = lenient_manager.enforce_limits(
            "X", DESIRED_SIZE, noisy_prices, positions
        )

        assert size == 0
        assert list(violations) == ["correlation_exceeded"]
        assert "with Y" in violations["correlation_exceeded"]
        corr = lenient_manager.position_correlations
        assert corr[("X", "Y")] == pytest.approx(1.0)
        assert corr[("Y", "X")] == pytest.approx(1.0)
        assert ("X", "Z") not in corr

    def test_does_not_mutate_callers_positions(self, lenient_manager, noisy_prices):
        positions = {"Y": {"value": 500.0}}
        size, violations = lenient_manager.enforce_limits(
            "X", DESIRED_SIZE, noisy_prices, positions
        )
        assert (size, violations) == (DESIRED_SIZE, {})
        assert positions == {"Y": {"value": 500.0}}


# =============================================================================
# get_risk_summary
# =============================================================================
class TestGetRiskSummary:
    @pytest.mark.parametrize("history", [None, [], [100.0]])
    def test_insufficient_history_is_invalid(self, risk_manager, history):
        summary = risk_manager.get_risk_summary("X", history)
        assert summary == {
            "symbol": "X",
            "valid": False,
            "error": "Insufficient price history",
        }

    def test_flat_prices_pass_every_check(self, risk_manager, flat_prices):
        summary = risk_manager.get_risk_summary("X", flat_prices)
        assert summary["valid"] is True
        for key in ("var_95", "es_95", "volatility", "max_drawdown"):
            assert summary[key]["value"] == 0
            assert summary[key]["passed"] is True
        assert summary["var_95"]["threshold"] == risk_manager.var_threshold
        assert summary["risk_score"] == 0

    def test_crash_series_fails_tail_checks(self, risk_manager, crash_prices):
        summary = risk_manager.get_risk_summary("X", crash_prices)
        assert summary["valid"] is True
        assert summary["var_95"]["passed"] is False
        assert summary["es_95"]["passed"] is False
        assert summary["max_drawdown"]["passed"] is False
        assert summary["max_drawdown"]["value"] == pytest.approx(
            abs(risk_manager._calculate_max_drawdown(crash_prices))
        )
        assert summary["risk_score"] == 1.0

    def test_non_numeric_history_returns_error(self, risk_manager):
        summary = risk_manager.get_risk_summary("X", ["a", "b"])
        assert summary["valid"] is False
        assert "could not convert" in summary["error"]


# =============================================================================
# Margin monitoring
# =============================================================================
class TestCalculateMarginRequirement:
    def test_values_and_liquidation_analysis(self, risk_manager):
        positions = {
            "AAPL": {"price": 200.0, "quantity": 10},  # value from qty * price
            "MSFT": {"value": 3000.0, "price": 300.0, "quantity": 5},  # explicit value wins
            "SHRT": {"price": 50.0, "quantity": -4},  # short: no liquidation price
        }
        result = risk_manager.calculate_margin_requirement(positions)

        total = 2000.0 + 3000.0 + (-200.0)
        assert result["total_position_value"] == pytest.approx(total)
        assert result["required_margin"] == pytest.approx(total * 0.25)
        assert result["margin_requirement_pct"] == 0.25

        aapl = result["position_analysis"]["AAPL"]
        assert aapl["value"] == 2000.0
        assert aapl["liquidation_price"] == pytest.approx(50.0)
        assert aapl["cushion_pct"] == pytest.approx(0.75)
        assert result["position_analysis"]["MSFT"]["value"] == 3000.0

        short = result["position_analysis"]["SHRT"]
        assert short["liquidation_price"] is None
        assert short["cushion_pct"] is None

    def test_empty_positions(self, risk_manager):
        result = risk_manager.calculate_margin_requirement({})
        assert result["total_position_value"] == 0
        assert result["required_margin"] == 0
        assert result["position_analysis"] == {}


class TestCheckMarginStatus:
    POSITIONS = {"SPY": {"price": 100.0, "quantity": 100}}  # $10,000 exposure

    @pytest.mark.parametrize(
        "equity, status, action, halt, warn",
        [
            (5000.0, "OK", "NONE", False, False),
            (3500.0, "OK", "NONE", False, False),  # exactly at warning: not below
            (3200.0, "WARNING", "REDUCE_EXPOSURE", False, True),
            (3000.0, "WARNING", "REDUCE_EXPOSURE", False, True),  # exactly at halt
            (2000.0, "CRITICAL", "HALT_TRADING", True, True),
        ],
    )
    def test_status_bands(self, risk_manager, equity, status, action, halt, warn):
        result = risk_manager.check_margin_status(equity, self.POSITIONS)
        assert result["margin_pct"] == pytest.approx(equity / 10_000)
        assert result["status"] == status
        assert result["action"] == action
        assert result["should_halt"] is halt
        assert result["should_warn"] is warn

    def test_deficit_and_reduction_arithmetic(self, risk_manager):
        result = risk_manager.check_margin_status(2000.0, self.POSITIONS)
        assert result["required_margin"] == pytest.approx(2500.0)
        assert result["margin_excess"] == pytest.approx(-500.0)
        assert result["margin_deficit"] == pytest.approx(500.0)
        # Safe exposure is equity / warning_threshold = 2000 / 0.35.
        assert result["reduction_needed"] == pytest.approx(10_000 - 2000 / 0.35)

    def test_healthy_account_needs_no_reduction(self, risk_manager):
        result = risk_manager.check_margin_status(8000.0, self.POSITIONS)
        assert result["margin_deficit"] == 0
        assert result["reduction_needed"] == 0

    def test_no_positions_is_fully_margined(self, risk_manager):
        result = risk_manager.check_margin_status(1000.0, {})
        assert result["margin_pct"] == 1.0
        assert result["status"] == "OK"
        assert result["should_halt"] is False

    def test_zero_warning_threshold_does_not_divide_by_zero(self, risk_manager):
        result = risk_manager.check_margin_status(
            2000.0, self.POSITIONS, warning_threshold=0.0, halt_threshold=0.0
        )
        assert result["status"] == "OK"
        assert result["reduction_needed"] == pytest.approx(10_000)


class TestLiquidationPrices:
    def test_empty_positions(self, risk_manager):
        assert risk_manager.get_liquidation_prices({}, equity=1000.0) == {}

    @pytest.mark.parametrize(
        "equity, expected_price",
        [
            (5000.0, 50.0),  # equity = 2x required margin: may fall 50%
            (2500.0, 100.0),  # exactly at requirement: liquidates now
            (1000.0, 100.0),  # under-margined: clamped at current price
            (0.0, 0.0),  # no equity: drop is 100%
        ],
    )
    def test_long_position_liquidation_price(self, risk_manager, equity, expected_price):
        positions = {"SPY": {"price": 100.0, "quantity": 100}}
        prices = risk_manager.get_liquidation_prices(positions, equity)
        assert prices == {"SPY": pytest.approx(expected_price)}

    def test_short_and_zero_quantity_positions_are_skipped(self, risk_manager):
        positions = {
            "SPY": {"price": 100.0, "quantity": 100},
            "SHRT": {"price": 50.0, "quantity": -10},
            "FLAT": {"price": 20.0, "quantity": 0, "value": 100.0},
        }
        prices = risk_manager.get_liquidation_prices(positions, equity=10_000.0)
        assert set(prices) == {"SPY"}

    def test_zero_margin_requirement_yields_zero(self, risk_manager):
        positions = {"SPY": {"price": 100.0, "quantity": 100}}
        prices = risk_manager.get_liquidation_prices(
            positions, equity=5000.0, broker_margin_requirement=0.0
        )
        assert prices == {"SPY": 0}


class TestShouldHaltForMargin:
    POSITIONS = {"SPY": {"value": 10_000.0}}

    def test_no_positions_never_halts(self, risk_manager):
        assert risk_manager.should_halt_for_margin(0.0, {}) is False

    @pytest.mark.parametrize(
        "equity, expected",
        [(2999.0, True), (3000.0, False), (9000.0, False)],
    )
    def test_halts_strictly_below_threshold(self, risk_manager, equity, expected):
        assert risk_manager.should_halt_for_margin(equity, self.POSITIONS) is expected

    def test_custom_threshold(self, risk_manager):
        assert risk_manager.should_halt_for_margin(4000.0, self.POSITIONS, halt_threshold=0.5)
