"""
Cross-Asset Data Providers - Fetch and process cross-asset signals.

This module implements providers for VIX term structure, yield curve,
and FX correlation data using yfinance (no API keys required).
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from data.alt_data_types import AltDataSource, SignalDirection
from data.alternative_data_provider import AltDataCache, AlternativeDataProvider
from data.cross_asset_types import (
    CrossAssetAggregatedSignal,
    CrossAssetSource,
    FxCorrelationSignal,
    VixTermStructureSignal,
    YieldCurveSignal,
)

logger = logging.getLogger(__name__)


class CrossAssetProvider(AlternativeDataProvider):
    """
    Base class for cross-asset data providers.

    Cross-asset signals apply globally (not per-symbol), so the symbol
    parameter is ignored in fetch_signal().
    """

    def __init__(
        self,
        source: AltDataSource,
        cache_ttl_seconds: int = 300,  # 5 min default cache
    ):
        super().__init__(source=source, cache_ttl_seconds=cache_ttl_seconds)
        self._yf = None  # Lazy-loaded yfinance

    def _get_yfinance(self):
        """Lazy-load yfinance to avoid import overhead."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.error("yfinance not installed. Run: pip install yfinance")
                raise
        return self._yf

    async def _fetch_ticker_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch ticker data from yfinance.

        Args:
            symbol: Ticker symbol (e.g., ^VIX, TLT, DX-Y.NYB)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y)
            interval: Data interval (1m, 5m, 1h, 1d)

        Returns:
            Dict with 'current', 'history', 'change_5d', 'change_20d'
        """
        try:
            yf = self._get_yfinance()
            ticker = yf.Ticker(symbol)

            # Fetch history
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            current = float(hist["Close"].iloc[-1])

            # Calculate changes
            change_5d = 0.0
            change_20d = 0.0
            std_20d = 1.0

            if len(hist) >= 5:
                price_5d_ago = float(hist["Close"].iloc[-5])
                change_5d = (current - price_5d_ago) / price_5d_ago

            if len(hist) >= 20:
                price_20d_ago = float(hist["Close"].iloc[-20])
                change_20d = (current - price_20d_ago) / price_20d_ago
                std_20d = float(hist["Close"].iloc[-20:].std())

            # Z-score
            mean_20d = float(hist["Close"].iloc[-20:].mean()) if len(hist) >= 20 else current
            zscore = (current - mean_20d) / std_20d if std_20d > 0 else 0

            return {
                "current": current,
                "history": hist,
                "change_5d": change_5d,
                "change_20d": change_20d,
                "std_20d": std_20d,
                "zscore": zscore,
            }

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None


class VixTermStructureProvider(CrossAssetProvider):
    """
    Provider for VIX term structure signals.

    Uses ^VIX (spot) and ^VIX3M (3-month) to calculate term structure.
    Contango = normal/bullish, Backwardation = fear/bearish.
    """

    # Symbols
    VIX_SPOT = "^VIX"
    VIX_3M = "^VIX3M"  # CBOE 3-month VIX index

    def __init__(self, cache_ttl_seconds: int = 300):
        # Use a pseudo AltDataSource for compatibility
        super().__init__(
            source=AltDataSource.NEWS_ADVANCED,  # Reuse existing enum
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._cross_asset_source = CrossAssetSource.VIX_STRUCTURE

    async def initialize(self) -> bool:
        """Initialize provider and verify data access."""
        try:
            # Test fetch VIX spot
            vix_data = await self._fetch_ticker_data(self.VIX_SPOT, period="5d")
            if vix_data is None:
                logger.warning("Could not fetch VIX data during initialization")
                return False

            self._initialized = True
            logger.info("VixTermStructureProvider initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize VixTermStructureProvider: {e}")
            return False

    async def fetch_signal(self, symbol: str = None) -> Optional[VixTermStructureSignal]:
        """
        Fetch VIX term structure signal.

        Args:
            symbol: Ignored - VIX applies globally

        Returns:
            VixTermStructureSignal or None if fetch fails
        """
        try:
            # Fetch VIX spot and 3-month
            vix_spot_data, vix_3m_data = await asyncio.gather(
                self._fetch_ticker_data(self.VIX_SPOT, period="1mo"),
                self._fetch_ticker_data(self.VIX_3M, period="1mo"),
            )

            if vix_spot_data is None:
                logger.warning("Could not fetch VIX spot data")
                return None

            vix_spot = vix_spot_data["current"]

            # Handle VIX3M - use spot + contango estimate if unavailable
            if vix_3m_data is not None:
                vix_3m = vix_3m_data["current"]
            else:
                # Typical contango is ~5% over 3 months in normal conditions
                vix_3m = vix_spot * 1.05
                logger.info("Using estimated VIX3M (VIX3M data unavailable)")

            # Calculate term structure
            term_slope = (vix_3m - vix_spot) / vix_spot if vix_spot > 0 else 0

            # Generate signal
            # Contango (positive slope) = bullish for equities (low fear)
            # Backwardation (negative slope) = bearish for equities (high fear)
            signal_value = term_slope  # Already normalized
            signal_value = max(-1.0, min(1.0, signal_value * 5))  # Scale for sensitivity

            # Confidence based on VIX level (higher VIX = more reliable signal)
            if vix_spot > 25:
                confidence = 0.8
            elif vix_spot > 20:
                confidence = 0.7
            elif vix_spot > 15:
                confidence = 0.6
            else:
                confidence = 0.5

            # Derive direction
            if signal_value > 0.1:
                direction = SignalDirection.BULLISH
            elif signal_value < -0.1:
                direction = SignalDirection.BEARISH
            else:
                direction = SignalDirection.NEUTRAL

            return VixTermStructureSignal(
                symbol="VIX",
                source=AltDataSource.NEWS_ADVANCED,
                timestamp=datetime.now(),
                signal_value=signal_value,
                confidence=confidence,
                direction=direction,
                vix_spot=vix_spot,
                vix_3m=vix_3m,
                term_slope=term_slope,
                vix_change_5d=vix_spot_data.get("change_5d", 0),
                vix_change_20d=vix_spot_data.get("change_20d", 0),
                raw_data={
                    "vix_spot": vix_spot,
                    "vix_3m": vix_3m,
                    "term_slope": term_slope,
                    "cross_asset_source": self._cross_asset_source.value,
                },
            )

        except Exception as e:
            logger.error(f"Error fetching VIX term structure: {e}")
            return None


class YieldCurveProvider(CrossAssetProvider):
    """
    Provider for yield curve signals using Treasury ETFs.

    Uses TLT (20Y), IEF (7-10Y), SHY (1-3Y) as proxies for yields.
    Rising TLT/SHY ratio indicates curve steepening.
    """

    # Treasury ETF symbols
    TLT = "TLT"  # 20+ Year Treasury
    IEF = "IEF"  # 7-10 Year Treasury
    SHY = "SHY"  # 1-3 Year Treasury

    def __init__(self, cache_ttl_seconds: int = 300):
        super().__init__(
            source=AltDataSource.NEWS_ADVANCED,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._cross_asset_source = CrossAssetSource.YIELD_CURVE

    async def initialize(self) -> bool:
        """Initialize provider and verify data access."""
        try:
            # Test fetch TLT
            tlt_data = await self._fetch_ticker_data(self.TLT, period="5d")
            if tlt_data is None:
                logger.warning("Could not fetch TLT data during initialization")
                return False

            self._initialized = True
            logger.info("YieldCurveProvider initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize YieldCurveProvider: {e}")
            return False

    async def fetch_signal(self, symbol: str = None) -> Optional[YieldCurveSignal]:
        """
        Fetch yield curve signal from Treasury ETFs.

        The curve slope is approximated by the ratio TLT/SHY change.
        When long-term rates rise faster (TLT falls more), curve steepens.

        Args:
            symbol: Ignored - yield curve applies globally

        Returns:
            YieldCurveSignal or None if fetch fails
        """
        try:
            # Fetch all Treasury ETFs
            tlt_data, ief_data, shy_data = await asyncio.gather(
                self._fetch_ticker_data(self.TLT, period="1mo"),
                self._fetch_ticker_data(self.IEF, period="1mo"),
                self._fetch_ticker_data(self.SHY, period="1mo"),
            )

            if tlt_data is None or shy_data is None:
                logger.warning("Could not fetch Treasury ETF data")
                return None

            tlt_price = tlt_data["current"]
            shy_price = shy_data["current"]
            ief_price = ief_data["current"] if ief_data else tlt_price * 0.95

            # Calculate curve slope proxy
            # Higher TLT/SHY ratio = steeper curve (long rates higher than short)
            # Note: ETF prices move inversely to yields
            # So TLT falling vs SHY = long rates rising = steepening
            slope_ratio = tlt_price / shy_price if shy_price > 0 else 1.0

            # Normalize to a sensible range
            # Typical TLT/SHY ratio is around 1.1-1.3
            # Center around 1.2, deviations indicate curve shape changes
            baseline_ratio = 1.2
            curve_slope = (slope_ratio - baseline_ratio) / baseline_ratio

            # Calculate changes
            tlt_change = tlt_data.get("change_5d", 0)
            shy_change = shy_data.get("change_5d", 0)
            slope_change_5d = tlt_change - shy_change  # Relative performance

            tlt_change_20d = tlt_data.get("change_20d", 0)
            shy_change_20d = shy_data.get("change_20d", 0)
            slope_change_20d = tlt_change_20d - shy_change_20d

            # Generate signal
            # Steepening (curve_slope rising) = growth expectations = bullish
            # Flattening/inverting = recession risk = bearish
            signal_value = np.tanh(curve_slope * 10)  # Normalize to [-1, 1]

            # Also consider the change direction
            if slope_change_5d > 0.01:  # Steepening
                signal_value = min(1.0, signal_value + 0.2)
            elif slope_change_5d < -0.01:  # Flattening
                signal_value = max(-1.0, signal_value - 0.2)

            # Confidence based on signal magnitude
            confidence = min(0.8, 0.5 + abs(curve_slope) * 3)

            # Derive direction
            if signal_value > 0.1:
                direction = SignalDirection.BULLISH
            elif signal_value < -0.1:
                direction = SignalDirection.BEARISH
            else:
                direction = SignalDirection.NEUTRAL

            return YieldCurveSignal(
                symbol="YIELD_CURVE",
                source=AltDataSource.NEWS_ADVANCED,
                timestamp=datetime.now(),
                signal_value=signal_value,
                confidence=confidence,
                direction=direction,
                short_rate_proxy=shy_price,
                mid_rate_proxy=ief_price,
                long_rate_proxy=tlt_price,
                curve_slope=curve_slope,
                curve_slope_change_5d=slope_change_5d,
                curve_slope_change_20d=slope_change_20d,
                is_inverted=curve_slope < -0.05,
                is_steepening=slope_change_5d > 0.01,
                is_flattening=slope_change_5d < -0.01,
                raw_data={
                    "tlt_price": tlt_price,
                    "ief_price": ief_price,
                    "shy_price": shy_price,
                    "slope_ratio": slope_ratio,
                    "cross_asset_source": self._cross_asset_source.value,
                },
            )

        except Exception as e:
            logger.error(f"Error fetching yield curve: {e}")
            return None


class FxCorrelationProvider(CrossAssetProvider):
    """
    Provider for FX correlation signals.

    Uses USD Index (DXY) and AUD/JPY as risk sentiment indicators.
    USD strength = risk-off, AUD/JPY strength = risk-on.
    """

    # FX symbols
    DXY = "DX-Y.NYB"  # USD Index
    AUDJPY = "AUDJPY=X"  # AUD/JPY cross

    def __init__(self, cache_ttl_seconds: int = 300):
        super().__init__(
            source=AltDataSource.NEWS_ADVANCED,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._cross_asset_source = CrossAssetSource.FX_CORRELATION

    async def initialize(self) -> bool:
        """Initialize provider and verify data access."""
        try:
            # Test fetch DXY
            dxy_data = await self._fetch_ticker_data(self.DXY, period="5d")
            if dxy_data is None:
                # Try backup - UUP ETF
                logger.info("DXY unavailable, trying UUP ETF")
                dxy_data = await self._fetch_ticker_data("UUP", period="5d")

            if dxy_data is None:
                logger.warning("Could not fetch USD data during initialization")
                return False

            self._initialized = True
            logger.info("FxCorrelationProvider initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FxCorrelationProvider: {e}")
            return False

    async def fetch_signal(self, symbol: str = None) -> Optional[FxCorrelationSignal]:
        """
        Fetch FX correlation signal.

        Combines USD Index and AUD/JPY for risk sentiment.

        Args:
            symbol: Ignored - FX correlation applies globally

        Returns:
            FxCorrelationSignal or None if fetch fails
        """
        try:
            # Fetch DXY and AUD/JPY
            dxy_data, audjpy_data = await asyncio.gather(
                self._fetch_ticker_data(self.DXY, period="1mo"),
                self._fetch_ticker_data(self.AUDJPY, period="1mo"),
            )

            # Fallback to UUP if DXY unavailable
            if dxy_data is None:
                dxy_data = await self._fetch_ticker_data("UUP", period="1mo")

            if dxy_data is None:
                logger.warning("Could not fetch USD data")
                return None

            dxy_level = dxy_data["current"]
            dxy_change_5d = dxy_data.get("change_5d", 0)
            dxy_change_20d = dxy_data.get("change_20d", 0)
            dxy_zscore = dxy_data.get("zscore", 0)

            # AUD/JPY data
            if audjpy_data:
                audjpy_level = audjpy_data["current"]
                audjpy_change_5d = audjpy_data.get("change_5d", 0)
                audjpy_change_20d = audjpy_data.get("change_20d", 0)
                audjpy_zscore = audjpy_data.get("zscore", 0)
            else:
                # Fallback - use only USD
                audjpy_level = 0
                audjpy_change_5d = 0
                audjpy_change_20d = 0
                audjpy_zscore = 0

            # Calculate risk appetite
            # USD strength (positive change) = risk-off (negative signal)
            # AUD/JPY strength (positive change) = risk-on (positive signal)
            usd_signal = -dxy_zscore  # Invert: USD weakness = risk-on
            audjpy_signal = audjpy_zscore  # AUD/JPY strength = risk-on

            # Weight: 50% USD, 50% AUD/JPY (if available)
            if audjpy_data:
                risk_appetite_score = 0.5 * usd_signal + 0.5 * audjpy_signal
            else:
                risk_appetite_score = usd_signal

            # Normalize to [-1, 1]
            risk_appetite_score = max(-1.0, min(1.0, risk_appetite_score))

            # Signal value
            signal_value = risk_appetite_score

            # Confidence based on signal agreement
            if audjpy_data:
                # If both signals agree, higher confidence
                usd_dir = -1 if dxy_change_5d > 0 else 1
                audjpy_dir = 1 if audjpy_change_5d > 0 else -1
                agreement = 1.0 if usd_dir == audjpy_dir else 0.5
                confidence = 0.5 + 0.3 * agreement
            else:
                confidence = 0.5

            # Derive direction
            if signal_value > 0.1:
                direction = SignalDirection.BULLISH
            elif signal_value < -0.1:
                direction = SignalDirection.BEARISH
            else:
                direction = SignalDirection.NEUTRAL

            return FxCorrelationSignal(
                symbol="FX_RISK",
                source=AltDataSource.NEWS_ADVANCED,
                timestamp=datetime.now(),
                signal_value=signal_value,
                confidence=confidence,
                direction=direction,
                dxy_level=dxy_level,
                dxy_change_5d=dxy_change_5d,
                dxy_change_20d=dxy_change_20d,
                dxy_zscore=dxy_zscore,
                audjpy_level=audjpy_level,
                audjpy_change_5d=audjpy_change_5d,
                audjpy_change_20d=audjpy_change_20d,
                audjpy_zscore=audjpy_zscore,
                risk_appetite_score=risk_appetite_score,
                raw_data={
                    "dxy_level": dxy_level,
                    "audjpy_level": audjpy_level,
                    "cross_asset_source": self._cross_asset_source.value,
                },
            )

        except Exception as e:
            logger.error(f"Error fetching FX correlation: {e}")
            return None


class CrossAssetAggregator:
    """
    Aggregates signals from all cross-asset providers.

    Provides a unified view of cross-asset risk sentiment.
    """

    def __init__(
        self,
        use_vix: bool = True,
        use_yield_curve: bool = True,
        use_fx: bool = True,
        cache_ttl_seconds: int = 300,
    ):
        self._use_vix = use_vix
        self._use_yield_curve = use_yield_curve
        self._use_fx = use_fx

        self._vix_provider = VixTermStructureProvider(cache_ttl_seconds) if use_vix else None
        self._yield_provider = YieldCurveProvider(cache_ttl_seconds) if use_yield_curve else None
        self._fx_provider = FxCorrelationProvider(cache_ttl_seconds) if use_fx else None

        self._cache = AltDataCache()
        self._cache_ttl = cache_ttl_seconds
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all providers."""
        results = []

        if self._vix_provider:
            results.append(await self._vix_provider.initialize())
        if self._yield_provider:
            results.append(await self._yield_provider.initialize())
        if self._fx_provider:
            results.append(await self._fx_provider.initialize())

        # Consider initialized if at least one provider works
        self._initialized = any(results)

        if self._initialized:
            logger.info(f"CrossAssetAggregator initialized: {sum(results)}/{len(results)} providers")
        else:
            logger.warning("CrossAssetAggregator: No providers initialized")

        return self._initialized

    async def get_signal(self) -> Optional[CrossAssetAggregatedSignal]:
        """
        Get aggregated cross-asset signal.

        Returns:
            CrossAssetAggregatedSignal with combined signals
        """
        if not self._initialized:
            await self.initialize()

        vix_signal = None
        yield_signal = None
        fx_signal = None

        # Fetch all signals in parallel
        tasks = []
        if self._vix_provider and self._vix_provider._initialized:
            tasks.append(("vix", self._vix_provider.fetch_signal()))
        if self._yield_provider and self._yield_provider._initialized:
            tasks.append(("yield", self._yield_provider.fetch_signal()))
        if self._fx_provider and self._fx_provider._initialized:
            tasks.append(("fx", self._fx_provider.fetch_signal()))

        if not tasks:
            logger.warning("No cross-asset providers available")
            return None

        # Execute in parallel
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for (name, _), result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {name} signal: {result}")
                continue

            if name == "vix":
                vix_signal = result
            elif name == "yield":
                yield_signal = result
            elif name == "fx":
                fx_signal = result

        # Create aggregated signal
        return CrossAssetAggregatedSignal(
            timestamp=datetime.now(),
            vix_signal=vix_signal,
            yield_curve_signal=yield_signal,
            fx_signal=fx_signal,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        return {
            "initialized": self._initialized,
            "vix_provider": self._vix_provider._initialized if self._vix_provider else False,
            "yield_provider": self._yield_provider._initialized if self._yield_provider else False,
            "fx_provider": self._fx_provider._initialized if self._fx_provider else False,
        }
