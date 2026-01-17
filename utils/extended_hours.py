"""
Extended Hours Trading Manager

Handles pre-market (4:00 AM - 9:30 AM EST) and after-hours (4:00 PM - 8:00 PM EST) trading.

Key Considerations:
- Lower liquidity (wider spreads, use limit orders)
- More volatility (news reactions)
- Different strategies (news-driven, gap trading)
- Risk management (smaller position sizes)

Usage:
    from utils.extended_hours import ExtendedHoursManager

    manager = ExtendedHoursManager(broker)

    # Check if we can trade now
    if manager.is_extended_hours():
        session = manager.get_current_session()  # 'pre_market' or 'after_hours'

        # Trade with extended hours settings
        await manager.execute_extended_hours_trade(symbol, side, quantity)
"""

import logging
from datetime import datetime, time
from typing import Optional, Dict, Tuple
import pytz

from brokers.order_builder import OrderBuilder

logger = logging.getLogger(__name__)


class ExtendedHoursManager:
    """
    Manager for extended hours trading (pre-market and after-hours).

    Market Hours (Eastern Time):
    - Pre-Market: 4:00 AM - 9:30 AM
    - Regular Hours: 9:30 AM - 4:00 PM
    - After-Hours: 4:00 PM - 8:00 PM
    """

    # Trading session times (Eastern Time)
    PRE_MARKET_START = time(4, 0)    # 4:00 AM
    PRE_MARKET_END = time(9, 30)     # 9:30 AM
    REGULAR_START = time(9, 30)      # 9:30 AM
    REGULAR_END = time(16, 0)        # 4:00 PM
    AFTER_HOURS_START = time(16, 0)  # 4:00 PM
    AFTER_HOURS_END = time(20, 0)    # 8:00 PM

    def __init__(self, broker, enable_pre_market: bool = True, enable_after_hours: bool = True):
        """
        Initialize extended hours manager.

        Args:
            broker: Trading broker instance
            enable_pre_market: Allow pre-market trading (default: True)
            enable_after_hours: Allow after-hours trading (default: True)
        """
        self.broker = broker
        self.enable_pre_market = enable_pre_market
        self.enable_after_hours = enable_after_hours

        # Extended hours trading parameters (more conservative)
        self.extended_hours_params = {
            'max_spread_pct': 0.005,      # Max 0.5% spread (tighter than regular)
            'position_size_multiplier': 0.5,  # 50% of regular position size
            'limit_order_offset_pct': 0.001,  # 0.1% from mid-price for limit orders
            'max_slippage_pct': 0.003,    # Max 0.3% slippage tolerance
            'min_volume': 10000,          # Minimum daily volume to trade
        }

        logger.info("ExtendedHoursManager initialized:")
        logger.info(f"  Pre-market: {'ENABLED' if enable_pre_market else 'DISABLED'}")
        logger.info(f"  After-hours: {'ENABLED' if enable_after_hours else 'DISABLED'}")

    def is_extended_hours(self) -> bool:
        """
        Check if current time is during extended hours.

        Returns:
            True if pre-market or after-hours
        """
        session = self.get_current_session()
        return session in ['pre_market', 'after_hours']

    def is_pre_market(self) -> bool:
        """Check if current time is pre-market."""
        return self.get_current_session() == 'pre_market'

    def is_after_hours(self) -> bool:
        """Check if current time is after-hours."""
        return self.get_current_session() == 'after_hours'

    def is_regular_hours(self) -> bool:
        """Check if current time is regular market hours."""
        return self.get_current_session() == 'regular'

    def get_current_session(self) -> str:
        """
        Get current market session.

        Returns:
            'pre_market', 'regular', 'after_hours', or 'closed'
        """
        # Get current time in Eastern Time
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern).time()

        if self.PRE_MARKET_START <= now < self.PRE_MARKET_END:
            return 'pre_market' if self.enable_pre_market else 'closed'
        elif self.REGULAR_START <= now < self.REGULAR_END:
            return 'regular'
        elif self.AFTER_HOURS_START <= now < self.AFTER_HOURS_END:
            return 'after_hours' if self.enable_after_hours else 'closed'
        else:
            return 'closed'

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
        if session not in ['pre_market', 'after_hours']:
            return False, f"Not in extended hours (current: {session})"

        # Check if symbol is eligible (most stocks are, but check anyway)
        try:
            # Get latest quote to check liquidity
            # Note: In real implementation, you'd check:
            # 1. Stock is not halted
            # 2. Sufficient liquidity in extended hours
            # 3. Not a penny stock
            # 4. Alpaca supports extended hours for this symbol

            return True, f"Eligible for {session} trading"

        except Exception as e:
            return False, f"Error checking eligibility: {e}"

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
                'symbol': symbol,
                'bid': 0.0,  # Replace with actual bid
                'ask': 0.0,  # Replace with actual ask
                'bid_size': 0,
                'ask_size': 0,
                'timestamp': datetime.now()
            }

            # Calculate spread
            if quote['bid'] > 0 and quote['ask'] > 0:
                spread = quote['ask'] - quote['bid']
                mid_price = (quote['bid'] + quote['ask']) / 2
                spread_pct = spread / mid_price if mid_price > 0 else 0

                quote['spread'] = spread
                quote['mid_price'] = mid_price
                quote['spread_pct'] = spread_pct

                # Check if spread is acceptable
                max_spread = self.extended_hours_params['max_spread_pct']
                quote['acceptable_spread'] = spread_pct <= max_spread

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
        multiplier = self.extended_hours_params['position_size_multiplier']
        adjusted = position_value * multiplier

        logger.debug(f"Extended hours position adjustment: ${position_value:.2f} -> ${adjusted:.2f} "
                    f"({multiplier:.0%} of regular)")

        return adjusted

    async def execute_extended_hours_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: str = 'limit'
    ) -> Optional[dict]:
        """
        Execute trade during extended hours with appropriate safeguards.

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            strategy: 'limit' (recommended) or 'market'

        Returns:
            Order result or None if failed
        """
        session = self.get_current_session()

        # Validate we're in extended hours
        if session not in ['pre_market', 'after_hours']:
            logger.warning(f"Cannot execute extended hours trade - not in extended hours (current: {session})")
            return None

        # Check symbol eligibility
        can_trade, reason = await self.can_trade_extended_hours(symbol)
        if not can_trade:
            logger.warning(f"Cannot trade {symbol} in extended hours: {reason}")
            return None

        try:
            # Get quote for price reference
            quote = await self.get_extended_hours_quote(symbol)

            if not quote:
                logger.error(f"Could not get quote for {symbol}")
                return None

            # Check spread
            if not quote.get('acceptable_spread', False):
                logger.warning(f"Spread too wide for {symbol}: {quote.get('spread_pct', 0):.2%}")
                return None

            # Use limit orders (safer for extended hours)
            if strategy == 'limit':
                # Calculate limit price based on mid-price
                mid_price = quote['mid_price']
                offset_pct = self.extended_hours_params['limit_order_offset_pct']

                if side == 'buy':
                    # Buy slightly above mid to increase fill probability
                    limit_price = mid_price * (1 + offset_pct)
                    # But don't exceed ask
                    limit_price = min(limit_price, quote['ask'])
                else:
                    # Sell slightly below mid
                    limit_price = mid_price * (1 - offset_pct)
                    # But don't go below bid
                    limit_price = max(limit_price, quote['bid'])

                logger.info(f"Extended hours {side.upper()} {symbol}:")
                logger.info(f"  Quantity: {quantity:.4f}")
                logger.info(f"  Limit Price: ${limit_price:.2f} (mid: ${mid_price:.2f})")
                logger.info(f"  Spread: {quote['spread_pct']:.2%}")
                logger.info(f"  Session: {session}")

                # Create limit order with extended hours flag
                order = (OrderBuilder(symbol, side, quantity)
                        .limit(limit_price)
                        .extended_hours()  # Enable extended hours trading
                        .day()
                        .build())

            else:
                # Market order (riskier in extended hours due to low liquidity)
                logger.warning(f"Using MARKET order in {session} for {symbol} - higher risk!")

                order = (OrderBuilder(symbol, side, quantity)
                        .market()
                        .extended_hours()
                        .day()
                        .build())

            # Submit order
            result = await self.broker.submit_order_advanced(order)

            if result:
                logger.info(f"✅ Extended hours {side.upper()} order submitted: {symbol} x {quantity:.4f}")
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
            session: 'pre_market' or 'after_hours'

        Returns:
            Dict with strategy recommendations
        """
        if session == 'pre_market':
            return {
                'primary': 'gap_trading',
                'description': 'Trade gaps from overnight news/earnings',
                'focus': 'Earnings announcements, news catalysts, futures direction',
                'risk': 'High - low liquidity, news-driven volatility',
                'best_symbols': 'Stocks with earnings or major news',
                'tips': [
                    'Use limit orders only',
                    'Wait for price stabilization',
                    'Monitor futures (ES, NQ) for direction',
                    'Focus on high-volume stocks',
                    'Reduce position size 50%'
                ]
            }
        elif session == 'after_hours':
            return {
                'primary': 'earnings_reaction',
                'description': 'Trade post-earnings moves',
                'focus': 'Earnings reports, guidance, analyst calls',
                'risk': 'High - emotional reactions, low liquidity',
                'best_symbols': 'Stocks that just reported earnings',
                'tips': [
                    'Wait 15-30 minutes after earnings release',
                    'Use limit orders with wide margins',
                    'Check options market for direction',
                    'Avoid first 5 minutes (most volatile)',
                    'Scale in/out of positions'
                ]
            }
        else:
            return {}

    async def get_extended_hours_opportunities(self) -> list:
        """
        Scan for trading opportunities during extended hours.

        Returns:
            List of opportunity dicts
        """
        session = self.get_current_session()

        if session not in ['pre_market', 'after_hours']:
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
            logger.info(f"Scanning for {session} opportunities...")

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
            #     'risk_reward': 2.5
            # })

        except Exception as e:
            logger.error(f"Error scanning extended hours opportunities: {e}")

        return opportunities

    def get_session_info(self) -> Dict:
        """
        Get detailed information about current session.

        Returns:
            Dict with session details
        """
        session = self.get_current_session()
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        info = {
            'session': session,
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'is_extended': session in ['pre_market', 'after_hours'],
            'can_trade': session in ['pre_market', 'after_hours', 'regular']
        }

        if session == 'pre_market':
            info.update({
                'session_name': 'Pre-Market',
                'start_time': '4:00 AM ET',
                'end_time': '9:30 AM ET',
                'recommended_strategy': 'Gap trading on news/earnings',
                'liquidity': 'Low',
                'volatility': 'High',
                'position_size_adj': '50% of regular'
            })
        elif session == 'after_hours':
            info.update({
                'session_name': 'After-Hours',
                'start_time': '4:00 PM ET',
                'end_time': '8:00 PM ET',
                'recommended_strategy': 'Earnings reactions',
                'liquidity': 'Low',
                'volatility': 'High',
                'position_size_adj': '50% of regular'
            })
        elif session == 'regular':
            info.update({
                'session_name': 'Regular Hours',
                'start_time': '9:30 AM ET',
                'end_time': '4:00 PM ET',
                'recommended_strategy': 'Standard strategies',
                'liquidity': 'Normal',
                'volatility': 'Normal',
                'position_size_adj': '100%'
            })

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
    output.append(f"📅 MARKET SESSION: {info['session_name']}")
    output.append("=" * 80)
    output.append(f"Current Time: {info['current_time']}")
    output.append(f"Session Hours: {info['start_time']} - {info['end_time']}")

    if info['is_extended']:
        output.append("\n⚠️  EXTENDED HOURS TRADING")
        output.append(f"Liquidity: {info['liquidity']} | Volatility: {info['volatility']}")
        output.append(f"Position Size: {info['position_size_adj']}")
        output.append(f"Strategy: {info['recommended_strategy']}")
    else:
        output.append("\n✅ Regular Market Hours")
        output.append("Full liquidity and normal strategies available")

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

    async def analyze_gap(self, symbol: str, prev_close: float, current_price: float) -> Optional[Dict]:
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
                'symbol': symbol,
                'signal': 'sell',  # Fade the gap
                'strategy': 'gap_fade',
                'gap_pct': gap_pct,
                'entry': current_price,
                'target': prev_close,  # Gap fill target
                'stop': current_price * 1.01,  # 1% stop
                'reason': f'Gap up {gap_pct:.1%} - fade opportunity'
            }
        else:
            # Gap down - look for bounce
            return {
                'symbol': symbol,
                'signal': 'buy',  # Buy the dip
                'strategy': 'gap_bounce',
                'gap_pct': gap_pct,
                'entry': current_price,
                'target': prev_close,  # Bounce target
                'stop': current_price * 0.99,  # 1% stop
                'reason': f'Gap down {gap_pct:.1%} - bounce opportunity'
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
        self,
        symbol: str,
        close_price: float,
        ah_price: float,
        earnings_beat: bool
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
                'symbol': symbol,
                'signal': 'buy',
                'strategy': 'earnings_continuation',
                'move_pct': move_pct,
                'entry': ah_price * 1.001,  # Slightly above current
                'target': ah_price * 1.03,  # 3% more upside
                'stop': close_price,  # Back to close
                'reason': f'Beat + {move_pct:.1%} move - continuation play'
            }
        # Earnings miss + negative move = fade (bounce)
        elif not earnings_beat and move_pct < 0:
            return {
                'symbol': symbol,
                'signal': 'buy',
                'strategy': 'earnings_oversold_bounce',
                'move_pct': move_pct,
                'entry': ah_price * 0.999,
                'target': ah_price * 1.02,  # 2% bounce
                'stop': ah_price * 0.98,  # 2% stop
                'reason': f'Miss + {move_pct:.1%} drop - oversold bounce'
            }

        return None
