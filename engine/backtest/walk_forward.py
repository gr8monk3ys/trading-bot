"""
BacktestEngine walk-forward validation mixin.

Owns:

    - ``run_walk_forward_backtest`` — splits the requested date range into
      ``n_folds`` train/test windows separated by an embargo period, runs
      ``run_backtest`` on each, and aggregates IS vs OOS Sharpe ratios to
      detect overfitting (OOS Sharpe < 50% of IS Sharpe).
    - ``_calculate_sharpe_from_equity`` — annualized Sharpe ratio from a
      raw equity curve, used both by the fold loop and directly by tests
      that monkeypatch it.

Depends on ``_fetch_trading_sessions_from_data_broker`` (core mixin) for
exchange-aware fold calendars and ``run_backtest`` (runner mixin) for the
per-fold execution.  This mixin is composed onto ``BacktestEngine`` along
with the other two.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


class BacktestWalkForwardMixin:
    """Walk-forward fold setup, train/test split, IS-vs-OOS aggregation."""

    async def run_walk_forward_backtest(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date,
        end_date,
        initial_capital: float = 100000,
        n_folds: int = 5,
        embargo_days: int = 5,
        train_pct: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Run a walk-forward backtest to detect overfitting.

        Walk-forward analysis splits data into multiple train/test periods,
        optimizes on training data, and validates on out-of-sample test data.
        This prevents curve-fitting and gives a realistic performance estimate.

        Args:
            strategy_class: Strategy class to instantiate and test
            symbols: List of symbols to trade
            start_date: Start date for backtest (date or datetime)
            end_date: End date for backtest (date or datetime)
            initial_capital: Starting capital per fold
            n_folds: Number of walk-forward folds
            embargo_days: Days to skip between train and test (prevents lookahead)
            train_pct: Percentage of each fold used for training (rest is test)

        Returns:
            Dictionary with walk-forward results including:
            - fold_results: Detailed results per fold
            - is_sharpe: Average in-sample Sharpe ratio
            - oos_sharpe: Average out-of-sample Sharpe ratio
            - degradation: Percentage degradation from IS to OOS
            - overfit_detected: True if OOS Sharpe < 50% of IS Sharpe
        """
        import numpy as np

        # Convert dates to datetime
        if hasattr(start_date, "strftime") and not hasattr(start_date, "hour"):
            start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            start_dt = start_date

        if hasattr(end_date, "strftime") and not hasattr(end_date, "hour"):
            end_dt = datetime.combine(end_date, datetime.min.time())
        else:
            end_dt = end_date

        logger.info(
            f"Running walk-forward backtest with {n_folds} folds, "
            f"{train_pct:.0%} train / {1 - train_pct:.0%} test split, "
            f"{embargo_days} day embargo"
        )

        from brokers.alpaca_broker import AlpacaBroker

        data_broker = self.broker if self.broker else AlpacaBroker(paper=True)
        trading_days = await self._fetch_trading_sessions_from_data_broker(
            data_broker,
            symbols,
            start_dt,
            end_dt,
        )

        total_days = len(trading_days)
        fold_size = total_days // n_folds

        logger.info(f"Total trading days: {total_days}, ~{fold_size} days per fold")

        # Results storage
        fold_results = []
        is_sharpes = []
        oos_sharpes = []
        is_returns = []
        oos_returns = []

        for fold_idx in range(n_folds):
            fold_start_idx = fold_idx * fold_size
            fold_end_idx = (
                min(fold_start_idx + fold_size, total_days)
                if fold_idx < n_folds - 1
                else total_days
            )

            # Split fold into train and test
            fold_days = trading_days[fold_start_idx:fold_end_idx]
            n_train = int(len(fold_days) * train_pct)

            train_days = fold_days[:n_train]
            # Add embargo period between train and test
            test_start_idx = n_train + embargo_days
            test_days = fold_days[test_start_idx:] if test_start_idx < len(fold_days) else []

            if len(train_days) < 20 or len(test_days) < 10:
                logger.warning(f"Fold {fold_idx + 1}: Insufficient data, skipping")
                continue

            train_start = train_days[0]
            train_end = train_days[-1]
            test_start = test_days[0]
            test_end = test_days[-1]

            logger.info(
                f"Fold {fold_idx + 1}/{n_folds}: "
                f"Train {train_start.date()} to {train_end.date()} ({len(train_days)} days), "
                f"Test {test_start.date()} to {test_end.date()} ({len(test_days)} days)"
            )

            # Run in-sample backtest
            try:
                is_result = await self.run_backtest(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=initial_capital,
                )
                is_equity = is_result.get("equity_curve", [initial_capital])
                is_return = (is_equity[-1] / is_equity[0]) - 1 if len(is_equity) > 1 else 0
                is_sharpe = self._calculate_sharpe_from_equity(is_equity)
            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} IS backtest failed: {e}")
                is_return = 0
                is_sharpe = 0
                is_result = {}

            # Run out-of-sample backtest
            try:
                oos_result = await self.run_backtest(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=initial_capital,
                )
                oos_equity = oos_result.get("equity_curve", [initial_capital])
                oos_return = (oos_equity[-1] / oos_equity[0]) - 1 if len(oos_equity) > 1 else 0
                oos_sharpe = self._calculate_sharpe_from_equity(oos_equity)
            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} OOS backtest failed: {e}")
                oos_return = 0
                oos_sharpe = 0
                oos_result = {}

            # Store fold results
            fold_result = {
                "fold": fold_idx + 1,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "train_days": len(train_days),
                "test_days": len(test_days),
                "is_return": is_return,
                "is_sharpe": is_sharpe,
                "oos_return": oos_return,
                "oos_sharpe": oos_sharpe,
                "is_trades": is_result.get("total_trades", 0),
                "oos_trades": oos_result.get("total_trades", 0),
            }
            fold_results.append(fold_result)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)
            is_returns.append(is_return)
            oos_returns.append(oos_return)

            logger.info(
                f"  IS: {is_return:+.2%} return, {is_sharpe:.2f} Sharpe | "
                f"OOS: {oos_return:+.2%} return, {oos_sharpe:.2f} Sharpe"
            )

        # Calculate aggregate metrics
        if not fold_results:
            logger.error("No valid folds completed")
            return {
                "fold_results": [],
                "is_sharpe": 0,
                "oos_sharpe": 0,
                "degradation": 0,
                "overfit_detected": True,
                "error": "No valid folds",
            }

        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
        avg_is_return = np.mean(is_returns) if is_returns else 0
        avg_oos_return = np.mean(oos_returns) if oos_returns else 0

        # Calculate degradation
        if avg_is_sharpe > 0:
            sharpe_degradation = 1 - (avg_oos_sharpe / avg_is_sharpe)
        else:
            sharpe_degradation = 0

        if avg_is_return > 0:
            return_degradation = 1 - (avg_oos_return / avg_is_return)
        else:
            return_degradation = 0

        # Detect overfitting: OOS Sharpe < 50% of IS Sharpe
        overfit_detected = avg_is_sharpe > 0 and avg_oos_sharpe < (avg_is_sharpe * 0.5)

        # Log summary
        logger.info(f"\n{'=' * 60}")
        logger.info("WALK-FORWARD BACKTEST RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Folds completed: {len(fold_results)}/{n_folds}")
        logger.info(f"Average IS Sharpe:  {avg_is_sharpe:.2f}")
        logger.info(f"Average OOS Sharpe: {avg_oos_sharpe:.2f}")
        logger.info(f"Sharpe Degradation: {sharpe_degradation:.1%}")
        logger.info(f"Average IS Return:  {avg_is_return:+.2%}")
        logger.info(f"Average OOS Return: {avg_oos_return:+.2%}")
        logger.info(f"Return Degradation: {return_degradation:.1%}")

        if overfit_detected:
            logger.warning(
                "⚠️ OVERFITTING DETECTED: OOS Sharpe < 50% of IS Sharpe. "
                "Strategy may be curve-fitted to historical data."
            )
        else:
            logger.info("✓ No significant overfitting detected")

        logger.info(f"{'=' * 60}\n")

        return {
            "fold_results": fold_results,
            "n_folds": n_folds,
            "embargo_days": embargo_days,
            "train_pct": train_pct,
            "is_sharpe": avg_is_sharpe,
            "oos_sharpe": avg_oos_sharpe,
            "is_return": avg_is_return,
            "oos_return": avg_oos_return,
            "sharpe_degradation": sharpe_degradation,
            "return_degradation": return_degradation,
            "overfit_detected": overfit_detected,
            "overfit_threshold": 0.5,  # OOS < 50% of IS = overfit
        }

    def _calculate_sharpe_from_equity(
        self, equity_curve: List[float], risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio from an equity curve.

        Args:
            equity_curve: List of daily portfolio values
            risk_free_rate: Annual risk-free rate (default 0)

        Returns:
            Annualized Sharpe ratio
        """
        import numpy as np

        if len(equity_curve) < 2:
            return 0.0

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        sharpe = np.mean(excess_returns) / np.std(excess_returns)

        # Annualize
        return sharpe * np.sqrt(252)
