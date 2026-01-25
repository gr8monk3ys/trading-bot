"""
Extended Hours Trading Manager

Handles extended and overnight trading sessions via Blue Ocean ATS.

Trading Sessions (Eastern Time):
- Pre-Market: 4:00 AM - 9:30 AM ET
- Regular Hours: 9:30 AM - 4:00 PM ET
- After-Hours: 4:00 PM - 8:00 PM ET
- Overnight: 8:00 PM - 4:00 AM ET (next day) - 24/5 via Blue Ocean ATS
  * Available Sunday 8:00 PM ET to Friday 4:00 AM ET

Key Considerations:
- Lower liquidity (wider spreads, use limit orders)
- More volatility (news reactions)
- Different strategies (news-driven, gap trading)
- Risk management (smaller position sizes)
- Overnight: Check symbol-specific overnight_tradeable status

Usage:
    from utils.extended_hours import ExtendedHoursManager, TradingSession

    manager = ExtendedHoursManager(broker, enable_overnight=True)

    # Get current session
    session = manager.get_current_session()  # Returns TradingSession enum

    # Check if we can trade now (includes overnight if enabled)
    if manager.is_market_open(include_overnight=True):
        await manager.execute_extended_hours_trade(symbol, side, quantity)

    # Check overnight tradability for specific symbol
    if await manager.check_overnight_tradeable(broker, "AAPL"):
        # Symbol supports overnight trading
        pass
"""

import logging
from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import pytz

from brokers.order_builder import OrderBuilder

if TYPE_CHECKING:
    from brokers.alpaca_broker import AlpacaBroker

logger = logging.getLogger(__name__)

# Eastern Time timezone constant
ET = pytz.timezone("America/New_York")


class TradingSession(Enum):
    """
    Enumeration of trading session types.

    Sessions represent different market hours with varying liquidity,
    volatility, and trading rules.
    """
    CLOSED = "closed"              # Market fully closed (weekend, holidays)
    PRE_MARKET = "pre_market"      # 4:00 AM - 9:30 AM ET
    REGULAR = "regular"            # 9:30 AM - 4:00 PM ET
    AFTER_HOURS = "after_hours"    # 4:00 PM - 8:00 PM ET
    OVERNIGHT = "overnight"        # 8:00 PM - 4:00 AM ET (Blue Ocean ATS)


class ExtendedHoursManager:
    """
    Manager for extended hours and overnight trading.

    Market Hours (Eastern Time):
    - Pre-Market: 4:00 AM - 9:30 AM
    - Regular Hours: 9:30 AM - 4:00 PM
    - After-Hours: 4:00 PM - 8:00 PM
    - Overnight: 8:00 PM - 4:00 AM (next day) - via Blue Ocean ATS

    24/5 Trading Schedule:
    - Opens: Sunday 8:00 PM ET
    - Closes: Friday 4:00 AM ET (end of overnight session)
    - Closed: Friday 4:00 AM - Sunday 8:00 PM ET
    """

    # Trading session times (Eastern Time)
    PRE_MARKET_START = time(4, 0)  # 4:00 AM
    PRE_MARKET_END = time(9, 30)  # 9:30 AM
    REGULAR_START = time(9, 30)  # 9:30 AM
    REGULAR_END = time(16, 0)  # 4:00 PM
    AFTER_HOURS_START = time(16, 0)  # 4:00 PM
    AFTER_HOURS_END = time(20, 0)  # 8:00 PM
    OVERNIGHT_START = time(20, 0)  # 8:00 PM
    OVERNIGHT_END = time(4, 0)  # 4:00 AM (next day)

    def __init__(
        self,
        broker=None,
        enable_pre_market: bool = True,
        enable_after_hours: bool = True,
        enable_overnight: bool = True,
    ):
        """
        Initialize extended hours manager.

        Args:
            broker: Trading broker instance (optional, can be set later)
            enable_pre_market: Allow pre-market trading (default: True)
            enable_after_hours: Allow after-hours trading (default: True)
            enable_overnight: Allow overnight trading via Blue Ocean ATS (default: True)
        """
        self.broker = broker
        self.enable_pre_market = enable_pre_market
        self.enable_after_hours = enable_after_hours
        self.enable_overnight = enable_overnight

        # Extended hours trading parameters (more conservative)
        self.extended_hours_params = {
            "max_spread_pct": 0.005,  # Max 0.5% spread (tighter than regular)
            "position_size_multiplier": 0.5,  # 50% of regular position size
            "limit_order_offset_pct": 0.001,  # 0.1% from mid-price for limit orders
            "max_slippage_pct": 0.003,  # Max 0.3% slippage tolerance
            "min_volume": 10000,  # Minimum daily volume to trade
        }

        # Overnight-specific parameters (even more conservative)
        self.overnight_params = {
            "position_size_multiplier": 0.3,  # 30% of regular position size
            "max_spread_pct": 0.01,  # Max 1.0% spread (overnight has wider spreads)
            "limit_order_offset_pct": 0.002,  # 0.2% from mid-price
            "max_slippage_pct": 0.005,  # Max 0.5% slippage tolerance
            "max_positions": 3,  # Limit concurrent overnight positions
        }

        logger.info("ExtendedHoursManager initialized:")
        logger.info(f"  Pre-market: {'ENABLED' if enable_pre_market else 'DISABLED'}")
        logger.info(f"  After-hours: {'ENABLED' if enable_after_hours else 'DISABLED'}")
        logger.info(f"  Overnight (Blue Ocean): {'ENABLED' if enable_overnight else 'DISABLED'}")

    def is_extended_hours(self) -> bool:
        """
        Check if current time is during extended hours (pre-market or after-hours).

        Note: Does not include overnight session. Use is_overnight() for that.

        Returns:
            True if pre-market or after-hours
        """
        session = self.get_current_session()
        return session in [TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS]

    def is_pre_market(self) -> bool:
        """Check if current time is pre-market."""
        return self.get_current_session() == TradingSession.PRE_MARKET

    def is_after_hours(self) -> bool:
        """Check if current time is after-hours."""
        return self.get_current_session() == TradingSession.AFTER_HOURS

    def is_regular_hours(self) -> bool:
        """Check if current time is regular market hours."""
        return self.get_current_session() == TradingSession.REGULAR

    def is_overnight(self) -> bool:
        """Check if current time is overnight session (Blue Ocean ATS)."""
        return self.get_current_session() == TradingSession.OVERNIGHT

    def get_current_session(self, dt: datetime = None) -> TradingSession:
        """
        Get current trading session.

        Args:
            dt: Datetime to check (default: now). Can be naive or timezone-aware.

        Returns:
            TradingSession enum value
        """
        # Handle datetime input
        if dt is None:
            dt = datetime.now(ET)
        elif dt.tzinfo is None:
            dt = ET.localize(dt)
        else:
            dt = dt.astimezone(ET)

        current_time = dt.time()
        weekday = dt.weekday()  # Monday = 0, Sunday = 6

        # Check if fully closed (Saturday or Friday overnight after cutoff)
        if weekday == 5:  # Saturday - fully closed
            return TradingSession.CLOSED

        # Sunday - only overnight session after 8 PM
        if weekday == 6:  # Sunday
            if current_time >= self.OVERNIGHT_START:
                return TradingSession.OVERNIGHT if self.enable_overnight else TradingSession.CLOSED
            return TradingSession.CLOSED

        # Friday special handling - overnight closes at 4 AM
        if weekday == 4:  # Friday
            # After 8 PM Friday = closed (no overnight into Saturday)
            if current_time >= self.AFTER_HOURS_END:
                return TradingSession.CLOSED

        # Check overnight session (8 PM previous day to 4 AM)
        # This applies to Mon-Fri before 4 AM (overnight from previous night)
        if current_time < self.OVERNIGHT_END:  # Before 4 AM
            return TradingSession.OVERNIGHT if self.enable_overnight else TradingSession.CLOSED

        # Pre-market: 4:00 AM - 9:30 AM
        if self.PRE_MARKET_START <= current_time < self.PRE_MARKET_END:
            return TradingSession.PRE_MARKET if self.enable_pre_market else TradingSession.CLOSED

        # Regular hours: 9:30 AM - 4:00 PM
        if self.REGULAR_START <= current_time < self.REGULAR_END:
            return TradingSession.REGULAR

        # After-hours: 4:00 PM - 8:00 PM
        if self.AFTER_HOURS_START <= current_time < self.AFTER_HOURS_END:
            return TradingSession.AFTER_HOURS if self.enable_after_hours else TradingSession.CLOSED

        # Overnight: 8:00 PM onwards (Mon-Thu only, Friday goes to CLOSED)
        if current_time >= self.OVERNIGHT_START:
            return TradingSession.OVERNIGHT if self.enable_overnight else TradingSession.CLOSED

        return TradingSession.CLOSED

    def is_market_open(
        self,
        include_extended: bool = True,
        include_overnight: bool = None,
    ) -> bool:
        """
        Check if any trading session is currently active.

        Args:
            include_extended: Include pre-market and after-hours (default: True)
            include_overnight: Include overnight session (default: follows enable_overnight)

        Returns:
            True if trading is possible in current session
        """
        if include_overnight is None:
            include_overnight = self.enable_overnight

        session = self.get_current_session()

        if session == TradingSession.REGULAR:
            return True
        if include_extended and session in (TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS):
            return True
        if include_overnight and session == TradingSession.OVERNIGHT:
            return True
        return False

    async def can_trade_extended_hours(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if a symbol can be traded during extended hours.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (can_trade, reason)
        """
        session = self.get_current_session()

        # Not in extended hours
        if session not in [TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS]:
            return False, f"Not in extended hours (current: {session.value})"

        # Check if symbol is eligible (most stocks are, but check anyway)
        try:
            # Get latest quote to check liquidity
            # Note: In real implementation, you'd check:
            # 1. Stock is not halted
            # 2. Sufficient liquidity in extended hours
            # 3. Not a penny stock
            # 4. Alpaca supports extended hours for this symbol

            return True, f"Eligible for {session.value} trading"

        except Exception as e:
            return False, f"Error checking eligibility: {e}"

    async def can_trade_overnight(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if a symbol can be traded during overnight session.

        Overnight trading requires:
        1. Current time is in overnight session
        2. Symbol has overnight_tradeable flag set
        3. Symbol is not halted for overnight trading

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (can_trade, reason)
        """
        session = self.get_current_session()

        # Not in overnight session
        if session != TradingSession.OVERNIGHT:
            return False, f"Not in overnight session (current: {session.value})"

        # Check if overnight is enabled
        if not self.enable_overnight:
            return False, "Overnight trading is disabled"

        # Check symbol-specific overnight tradability
        if self.broker is not None:
            is_tradeable = await self.check_overnight_tradeable(self.broker, symbol)
            if not is_tradeable:
                return False, f"{symbol} is not available for overnight trading"

        return True, f"Eligible for overnight trading"

    async def check_overnight_tradeable(self, broker: "AlpacaBroker", symbol: str) -> bool:
        """
        Check if a symbol can be traded in overnight session via Blue Ocean ATS.

        Args:
            broker: AlpacaBroker instance
            symbol: Stock symbol

        Returns:
            True if overnight trading is available for this symbol
        """
        try:
            asset = await broker.get_asset(symbol)
            if asset:
                # Check overnight_tradeable flag and ensure not halted
                overnight_tradeable = asset.get("overnight_tradeable", False)
                overnight_halted = asset.get("overnight_halted", True)

                is_available = overnight_tradeable and not overnight_halted

                if not is_available:
                    logger.debug(
                        f"{symbol} overnight status: tradeable={overnight_tradeable}, "
                        f"halted={overnight_halted}"
                    )

                return is_available
            return False
        except Exception as e:
            logger.warning(f"Could not check overnight status for {symbol}: {e}")
            return False

    async def get_extended_hours_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get quote during extended hours with spread analysis.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with bid, ask, spread info or None
        """
        try:
            # Get latest quote from broker
            # In production, use: quote = await self.broker.get_latest_quote(symbol)
            # For now, this is a placeholder

            quote = {
                "symbol": symbol,
                "bid": 0.0,  # Replace with actual bid
                "ask": 0.0,  # Replace with actual ask
                "bid_size": 0,
                "ask_size": 0,
                "timestamp": datetime.now(),
            }

            # Calculate spread
            if quote["bid"] > 0 and quote["ask"] > 0:
                spread = quote["ask"] - quote["bid"]
                mid_price = (quote["bid"] + quote["ask"]) / 2
                spread_pct = spread / mid_price if mid_price > 0 else 0

                quote["spread"] = spread
                quote["mid_price"] = mid_price
                quote["spread_pct"] = spread_pct

                # Check if spread is acceptable
                max_spread = self.extended_hours_params["max_spread_pct"]
                quote["acceptable_spread"] = spread_pct <= max_spread

            return quote

        except Exception as e:
            logger.error(f"Error getting extended hours quote for {symbol}: {e}")
            return None

    def adjust_position_size_for_extended_hours(self, position_value: float) -> float:
        """
        Reduce position size for extended hours trading.

        Args:
            position_value: Regular hours position value

        Returns:
            Adjusted position value (smaller for extended hours)
        """
        multiplier = self.extended_hours_params["position_size_multiplier"]
        adjusted = position_value * multiplier

        logger.debug(
            f"Extended hours position adjustment: ${position_value:.2f} -> ${adjusted:.2f} "
            f"({multiplier:.0%} of regular)"
        )

        return adjusted

    def adjust_position_size_for_overnight(self, position_value: float) -> float:
        """
        Reduce position size for overnight trading (more aggressive reduction).

        Overnight trading has lower liquidity and wider spreads than
        regular extended hours, so positions should be even smaller.

        Args:
            position_value: Regular hours position value

        Returns:
            Adjusted position value (30% of regular by default)
        """
        multiplier = self.overnight_params["position_size_multiplier"]
        adjusted = position_value * multiplier

        logger.debug(
            f"Overnight position adjustment: ${position_value:.2f} -> ${adjusted:.2f} "
            f"({multiplier:.0%} of regular)"
        )

        return adjusted

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier for the current session.

        Returns:
            Multiplier to apply to regular position size (0.3 - 1.0)
        """
        session = self.get_current_session()

        if session == TradingSession.REGULAR:
            return 1.0
        elif session in [TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS]:
            return self.extended_hours_params["position_size_multiplier"]
        elif session == TradingSession.OVERNIGHT:
            return self.overnight_params["position_size_multiplier"]
        else:  # CLOSED
            return 0.0

    async def execute_extended_hours_trade(
        self, symbol: str, side: str, quantity: float, strategy: str = "limit"
    ) -> Optional[dict]:
        """
        Execute trade during extended hours or overnight session with appropriate safeguards.

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            strategy: 'limit' (recommended) or 'market'

        Returns:
            Order result or None if failed
        """
        session = self.get_current_session()

        # Validate we're in an eligible session
        valid_sessions = [TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS]
        if self.enable_overnight:
            valid_sessions.append(TradingSession.OVERNIGHT)

        if session not in valid_sessions:
            logger.warning(
                f"Cannot execute extended hours trade - not in extended hours "
                f"(current: {session.value})"
            )
            return None

        # Check symbol eligibility based on session type
        if session == TradingSession.OVERNIGHT:
            can_trade, reason = await self.can_trade_overnight(symbol)
        else:
            can_trade, reason = await self.can_trade_extended_hours(symbol)

        if not can_trade:
            logger.warning(f"Cannot trade {symbol} in {session.value}: {reason}")
            return None

        try:
            # Get quote for price reference
            quote = await self.get_extended_hours_quote(symbol)

            if not quote:
                logger.error(f"Could not get quote for {symbol}")
                return None

            # Check spread
            if not quote.get("acceptable_spread", False):
                logger.warning(f"Spread too wide for {symbol}: {quote.get('spread_pct', 0):.2%}")
                return None

            # Use limit orders (safer for extended hours)
            if strategy == "limit":
                # Calculate limit price based on mid-price
                mid_price = quote["mid_price"]
                offset_pct = self.extended_hours_params["limit_order_offset_pct"]

                if side == "buy":
                    # Buy slightly above mid to increase fill probability
                    limit_price = mid_price * (1 + offset_pct)
                    # But don't exceed ask
                    limit_price = min(limit_price, quote["ask"])
                else:
                    # Sell slightly below mid
                    limit_price = mid_price * (1 - offset_pct)
                    # But don't go below bid
                    limit_price = max(limit_price, quote["bid"])

                logger.info(f"Extended hours {side.upper()} {symbol}:")
                logger.info(f"  Quantity: {quantity:.4f}")
                logger.info(f"  Limit Price: ${limit_price:.2f} (mid: ${mid_price:.2f})")
                logger.info(f"  Spread: {quote['spread_pct']:.2%}")
                logger.info(f"  Session: {session}")

                # Create limit order with extended hours flag
                order = (
                    OrderBuilder(symbol, side, quantity)
                    .limit(limit_price)
                    .extended_hours()  # Enable extended hours trading
                    .day()
                    .build()
                )

            else:
                # Market order (riskier in extended hours due to low liquidity)
                logger.warning(f"Using MARKET order in {session} for {symbol} - higher risk!")

                order = OrderBuilder(symbol, side, quantity).market().extended_hours().day().build()

            # Submit order
            result = await self.broker.submit_order_advanced(order)

            if result:
                logger.info(
                    f"✅ Extended hours {side.upper()} order submitted: {symbol} x {quantity:.4f}"
                )
            else:
                logger.error(f"❌ Extended hours order failed: {symbol}")

            return result

        except Exception as e:
            logger.error(f"Error executing extended hours trade for {symbol}: {e}", exc_info=True)
            return None

    def get_extended_hours_strategies(self, session: str) -> Dict[str, str]:
        """
        Get recommended strategies for each extended hours session.

        Args:
            session: 'pre_market', 'after_hours', or 'overnight'
                     (accepts TradingSession enum values or string names)

        Returns:
            Dict with strategy recommendations
        """
        # Handle TradingSession enum
        if isinstance(session, TradingSession):
            session = session.value

        if session == "pre_market":
            return {
                "primary": "gap_trading",
                "description": "Trade gaps from overnight news/earnings",
                "focus": "Earnings announcements, news catalysts, futures direction",
                "risk": "High - low liquidity, news-driven volatility",
                "best_symbols": "Stocks with earnings or major news",
                "tips": [
                    "Use limit orders only",
                    "Wait for price stabilization",
                    "Monitor futures (ES, NQ) for direction",
                    "Focus on high-volume stocks",
                    "Reduce position size 50%",
                ],
            }
        elif session == "after_hours":
            return {
                "primary": "earnings_reaction",
                "description": "Trade post-earnings moves",
                "focus": "Earnings reports, guidance, analyst calls",
                "risk": "High - emotional reactions, low liquidity",
                "best_symbols": "Stocks that just reported earnings",
                "tips": [
                    "Wait 15-30 minutes after earnings release",
                    "Use limit orders with wide margins",
                    "Check options market for direction",
                    "Avoid first 5 minutes (most volatile)",
                    "Scale in/out of positions",
                ],
            }
        elif session == "overnight":
            return {
                "primary": "low_risk_positioning",
                "description": "Position for next day with low-risk overnight trades",
                "focus": "Index-following stocks, ETFs, anticipating next-day news",
                "risk": "Medium - very low liquidity, wide spreads",
                "best_symbols": "Blue-chip stocks, major ETFs (SPY, QQQ) if overnight-enabled",
                "venue": "Blue Ocean ATS",
                "tips": [
                    "Only trade overnight-enabled symbols",
                    "Use limit orders ONLY (never market orders)",
                    "Reduce position size to 30% of regular",
                    "Monitor international markets for direction",
                    "Set wider stop-losses due to low liquidity",
                    "Check symbol overnight_tradeable flag before trading",
                    "Avoid illiquid symbols even if overnight-enabled",
                    "Consider timezone - Asian/European market movements",
                ],
            }
        else:
            return {}

    async def get_extended_hours_opportunities(self) -> list:
        """
        Scan for trading opportunities during extended hours or overnight.

        Returns:
            List of opportunity dicts
        """
        session = self.get_current_session()

        valid_sessions = [TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS]
        if self.enable_overnight:
            valid_sessions.append(TradingSession.OVERNIGHT)

        if session not in valid_sessions:
            return []

        opportunities = []

        try:
            # In production, this would:
            # 1. Scan for stocks with news/earnings
            # 2. Check for significant gaps
            # 3. Analyze volume and liquidity
            # 4. Calculate risk/reward
            # 5. Return ranked opportunities

            # Placeholder for now
            logger.info(f"Scanning for {session.value} opportunities...")

            # Example opportunity structure:
            # opportunities.append({
            #     'symbol': 'AAPL',
            #     'catalyst': 'Earnings beat',
            #     'gap_pct': 0.03,  # 3% gap
            #     'liquidity': 'good',
            #     'strategy': 'gap_fade',
            #     'entry_price': 175.50,
            #     'target_price': 173.00,
            #     'stop_price': 177.00,
            #     'risk_reward': 2.5,
            #     'session': session.value,  # Track which session
            #     'overnight_tradeable': True,  # For overnight session
            # })

        except Exception as e:
            logger.error(f"Error scanning {session.value} opportunities: {e}")

        return opportunities

    def get_session_info(self, dt: datetime = None) -> Dict:
        """
        Get detailed information about current session.

        Args:
            dt: Optional datetime to check (default: now)

        Returns:
            Dict with session details including:
            - session: TradingSession enum value
            - session_name: Human-readable session name
            - current_time_et: Current time in ET
            - day_of_week: Current day name
            - is_regular_hours: True if regular market hours
            - is_extended_hours: True if pre-market or after-hours
            - is_overnight: True if overnight session
            - can_trade: True if any trading is possible
            - Additional session-specific info (times, strategies, etc.)
        """
        session = self.get_current_session(dt)

        if dt is None:
            now = datetime.now(ET)
        elif dt.tzinfo is None:
            now = ET.localize(dt)
        else:
            now = dt.astimezone(ET)

        info = {
            "session": session.value,
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "current_time_et": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "is_regular_hours": session == TradingSession.REGULAR,
            "is_extended": session in [TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS],
            "is_overnight": session == TradingSession.OVERNIGHT,
            "can_trade": self.is_market_open(),
        }

        if session == TradingSession.PRE_MARKET:
            info.update(
                {
                    "session_name": "Pre-Market",
                    "start_time": "4:00 AM ET",
                    "end_time": "9:30 AM ET",
                    "recommended_strategy": "Gap trading on news/earnings",
                    "liquidity": "Low",
                    "volatility": "High",
                    "position_size_adj": "50% of regular",
                }
            )
        elif session == TradingSession.AFTER_HOURS:
            info.update(
                {
                    "session_name": "After-Hours",
                    "start_time": "4:00 PM ET",
                    "end_time": "8:00 PM ET",
                    "recommended_strategy": "Earnings reactions",
                    "liquidity": "Low",
                    "volatility": "High",
                    "position_size_adj": "50% of regular",
                }
            )
        elif session == TradingSession.REGULAR:
            info.update(
                {
                    "session_name": "Regular Hours",
                    "start_time": "9:30 AM ET",
                    "end_time": "4:00 PM ET",
                    "recommended_strategy": "Standard strategies",
                    "liquidity": "Normal",
                    "volatility": "Normal",
                    "position_size_adj": "100%",
                }
            )
        elif session == TradingSession.OVERNIGHT:
            info.update(
                {
                    "session_name": "Overnight (Blue Ocean ATS)",
                    "start_time": "8:00 PM ET",
                    "end_time": "4:00 AM ET",
                    "recommended_strategy": "Low-risk positions, news anticipation",
                    "liquidity": "Very Low",
                    "volatility": "Low-Medium",
                    "position_size_adj": "30% of regular",
                    "notes": [
                        "Trades execute via Blue Ocean ATS",
                        "Check symbol overnight_tradeable status",
                        "Wider spreads than daytime sessions",
                        "Limited to overnight-enabled symbols",
                    ],
                }
            )
        else:  # CLOSED
            info.update(
                {
                    "session_name": "Market Closed",
                    "recommended_strategy": "Wait for market open",
                    "notes": [
                        "No trading available",
                        "Weekend: Saturday all day, Sunday before 8 PM ET",
                    ],
                }
            )

        return info


# ==================== HELPER FUNCTIONS ====================


def format_session_info(info: Dict) -> str:
    """
    Format session info for display.

    Args:
        info: Session info dict from get_session_info()

    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 80)
    output.append(f"MARKET SESSION: {info['session_name']}")
    output.append("=" * 80)
    output.append(f"Current Time: {info['current_time']}")

    if "start_time" in info and "end_time" in info:
        output.append(f"Session Hours: {info['start_time']} - {info['end_time']}")

    if info.get("is_overnight"):
        output.append("\nOVERNIGHT TRADING (Blue Ocean ATS)")
        output.append(f"Liquidity: {info.get('liquidity', 'Very Low')} | Volatility: {info.get('volatility', 'Low-Medium')}")
        output.append(f"Position Size: {info.get('position_size_adj', '30% of regular')}")
        output.append(f"Strategy: {info.get('recommended_strategy', 'Low-risk positioning')}")
        if "notes" in info:
            output.append("\nImportant Notes:")
            for note in info["notes"]:
                output.append(f"  - {note}")
    elif info.get("is_extended"):
        output.append("\nEXTENDED HOURS TRADING")
        output.append(f"Liquidity: {info['liquidity']} | Volatility: {info['volatility']}")
        output.append(f"Position Size: {info['position_size_adj']}")
        output.append(f"Strategy: {info['recommended_strategy']}")
    elif info.get("is_regular_hours"):
        output.append("\nRegular Market Hours")
        output.append("Full liquidity and normal strategies available")
    else:
        output.append("\nMarket Closed")
        if "notes" in info:
            for note in info["notes"]:
                output.append(f"  - {note}")

    output.append("=" * 80)

    return "\n".join(output)


# ==================== EXTENDED HOURS STRATEGIES ====================


class GapTradingStrategy:
    """
    Gap trading strategy for pre-market.

    Trades gaps created by overnight news, earnings, or futures moves.
    """

    def __init__(self, gap_threshold: float = 0.02):
        """
        Initialize gap trading strategy.

        Args:
            gap_threshold: Minimum gap percentage to trade (default: 2%)
        """
        self.gap_threshold = gap_threshold

    async def analyze_gap(
        self, symbol: str, prev_close: float, current_price: float
    ) -> Optional[Dict]:
        """
        Analyze gap and return trade signal.

        Args:
            symbol: Stock symbol
            prev_close: Previous day's close
            current_price: Current pre-market price

        Returns:
            Trade signal dict or None
        """
        gap_pct = (current_price - prev_close) / prev_close

        if abs(gap_pct) < self.gap_threshold:
            return None

        # Gap up
        if gap_pct > 0:
            # Look for gap fill opportunities (fade the gap)
            return {
                "symbol": symbol,
                "signal": "sell",  # Fade the gap
                "strategy": "gap_fade",
                "gap_pct": gap_pct,
                "entry": current_price,
                "target": prev_close,  # Gap fill target
                "stop": current_price * 1.01,  # 1% stop
                "reason": f"Gap up {gap_pct:.1%} - fade opportunity",
            }
        else:
            # Gap down - look for bounce
            return {
                "symbol": symbol,
                "signal": "buy",  # Buy the dip
                "strategy": "gap_bounce",
                "gap_pct": gap_pct,
                "entry": current_price,
                "target": prev_close,  # Bounce target
                "stop": current_price * 0.99,  # 1% stop
                "reason": f"Gap down {gap_pct:.1%} - bounce opportunity",
            }


class EarningsReactionStrategy:
    """
    After-hours earnings reaction trading strategy.
    """

    def __init__(self, min_move_pct: float = 0.03):
        """
        Initialize earnings reaction strategy.

        Args:
            min_move_pct: Minimum earnings move to trade (default: 3%)
        """
        self.min_move_pct = min_move_pct

    async def analyze_earnings_move(
        self, symbol: str, close_price: float, ah_price: float, earnings_beat: bool
    ) -> Optional[Dict]:
        """
        Analyze post-earnings move.

        Args:
            symbol: Stock symbol
            close_price: Close price before earnings
            ah_price: After-hours price
            earnings_beat: Did company beat estimates?

        Returns:
            Trade signal dict or None
        """
        move_pct = (ah_price - close_price) / close_price

        if abs(move_pct) < self.min_move_pct:
            return None

        # Earnings beat + positive move = continuation
        if earnings_beat and move_pct > 0:
            return {
                "symbol": symbol,
                "signal": "buy",
                "strategy": "earnings_continuation",
                "move_pct": move_pct,
                "entry": ah_price * 1.001,  # Slightly above current
                "target": ah_price * 1.03,  # 3% more upside
                "stop": close_price,  # Back to close
                "reason": f"Beat + {move_pct:.1%} move - continuation play",
            }
        # Earnings miss + negative move = fade (bounce)
        elif not earnings_beat and move_pct < 0:
            return {
                "symbol": symbol,
                "signal": "buy",
                "strategy": "earnings_oversold_bounce",
                "move_pct": move_pct,
                "entry": ah_price * 0.999,
                "target": ah_price * 1.02,  # 2% bounce
                "stop": ah_price * 0.98,  # 2% stop
                "reason": f"Miss + {move_pct:.1%} drop - oversold bounce",
            }

        return None
