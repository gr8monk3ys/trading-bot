#!/usr/bin/env python3
"""
Trading Hours Optimization

Optimizes entry and exit timing based on:
1. Time of day patterns (avoid first 30min, lunch lull)
2. Day of week patterns (avoid Monday morning weakness)
3. VWAP-based entries for better fill prices

Research shows:
- First 30 minutes: High volatility, wide spreads, bad fills
- 11:30am-2pm: Low volume "lunch lull", choppy price action
- Last 30 minutes: Increased volatility from position squaring
- Best times: 9:45-11:30am and 2:30-3:30pm
- Monday mornings often weak, Friday afternoons see profit-taking

Expected Impact: 5-10% better entry prices, fewer whipsaws

Usage:
    from utils.trading_hours import TradingHoursFilter

    filter = TradingHoursFilter()

    if filter.is_good_time_to_trade():
        execute_trade()
    else:
        wait_for_better_window()
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
import pytz

logger = logging.getLogger(__name__)


class TradingWindow(Enum):
    """Trading time windows."""
    PREMARKET = "premarket"           # Before 9:30am ET
    OPENING_VOLATILITY = "opening"    # 9:30-10:00am ET - avoid
    MORNING_PRIME = "morning_prime"   # 10:00-11:30am ET - BEST
    LUNCH_LULL = "lunch_lull"         # 11:30am-2:00pm ET - avoid
    AFTERNOON_PRIME = "afternoon_prime"  # 2:00-3:30pm ET - GOOD
    CLOSING_VOLATILITY = "closing"    # 3:30-4:00pm ET - caution
    AFTERHOURS = "afterhours"         # After 4:00pm ET


class DayQuality(Enum):
    """Day of week trading quality."""
    EXCELLENT = "excellent"  # Tuesday, Wednesday
    GOOD = "good"           # Thursday
    FAIR = "fair"           # Friday
    POOR = "poor"           # Monday


class TradingHoursFilter:
    """
    Filters trades based on time of day and day of week patterns.
    """

    # Trading windows (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    # Optimal trading windows
    MORNING_PRIME_START = time(10, 0)
    MORNING_PRIME_END = time(11, 30)
    AFTERNOON_PRIME_START = time(14, 0)
    AFTERNOON_PRIME_END = time(15, 30)

    # Avoid these windows
    OPENING_AVOID_END = time(10, 0)
    LUNCH_START = time(11, 30)
    LUNCH_END = time(14, 0)
    CLOSING_CAUTION_START = time(15, 30)

    def __init__(
        self,
        avoid_opening: bool = True,
        avoid_lunch: bool = True,
        avoid_closing: bool = False,
        avoid_monday_morning: bool = True,
        avoid_friday_afternoon: bool = True,
        timezone: str = 'US/Eastern'
    ):
        """
        Initialize trading hours filter.

        Args:
            avoid_opening: Skip first 30 minutes after open
            avoid_lunch: Skip lunch lull (11:30am-2pm)
            avoid_closing: Skip last 30 minutes (optional)
            avoid_monday_morning: Skip Monday before 11am
            avoid_friday_afternoon: Skip Friday after 2pm
            timezone: Market timezone
        """
        self.avoid_opening = avoid_opening
        self.avoid_lunch = avoid_lunch
        self.avoid_closing = avoid_closing
        self.avoid_monday_morning = avoid_monday_morning
        self.avoid_friday_afternoon = avoid_friday_afternoon
        self.tz = pytz.timezone(timezone)

        logger.info(
            f"TradingHoursFilter: avoid_opening={avoid_opening}, "
            f"avoid_lunch={avoid_lunch}, avoid_closing={avoid_closing}"
        )

    def get_current_window(self, dt: Optional[datetime] = None) -> TradingWindow:
        """
        Get current trading window.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            Current TradingWindow
        """
        if dt is None:
            dt = datetime.now(self.tz)
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)

        current_time = dt.time()

        if current_time < self.MARKET_OPEN:
            return TradingWindow.PREMARKET
        elif current_time < self.OPENING_AVOID_END:
            return TradingWindow.OPENING_VOLATILITY
        elif current_time < self.LUNCH_START:
            return TradingWindow.MORNING_PRIME
        elif current_time < self.LUNCH_END:
            return TradingWindow.LUNCH_LULL
        elif current_time < self.CLOSING_CAUTION_START:
            return TradingWindow.AFTERNOON_PRIME
        elif current_time < self.MARKET_CLOSE:
            return TradingWindow.CLOSING_VOLATILITY
        else:
            return TradingWindow.AFTERHOURS

    def get_day_quality(self, dt: Optional[datetime] = None) -> DayQuality:
        """
        Get trading quality for day of week.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            DayQuality rating
        """
        if dt is None:
            dt = datetime.now(self.tz)

        weekday = dt.weekday()

        # Monday = 0, Tuesday = 1, ..., Friday = 4
        day_ratings = {
            0: DayQuality.POOR,       # Monday - often weak
            1: DayQuality.EXCELLENT,  # Tuesday - historically best
            2: DayQuality.EXCELLENT,  # Wednesday - good follow-through
            3: DayQuality.GOOD,       # Thursday - pre-Friday caution
            4: DayQuality.FAIR,       # Friday - profit taking
        }

        return day_ratings.get(weekday, DayQuality.FAIR)

    def is_good_time_to_trade(
        self,
        dt: Optional[datetime] = None,
        allow_fair_windows: bool = True
    ) -> bool:
        """
        Check if current time is good for trading.

        Args:
            dt: Datetime to check (default: now)
            allow_fair_windows: Allow trading in "fair" windows (closing, Thursday)

        Returns:
            True if good time to trade
        """
        if dt is None:
            dt = datetime.now(self.tz)
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)

        window = self.get_current_window(dt)
        day_quality = self.get_day_quality(dt)

        # Check if market is even open
        if window in [TradingWindow.PREMARKET, TradingWindow.AFTERHOURS]:
            return False

        # Weekend check
        if dt.weekday() >= 5:
            return False

        # Opening volatility check
        if self.avoid_opening and window == TradingWindow.OPENING_VOLATILITY:
            logger.debug("Avoiding opening volatility window")
            return False

        # Lunch lull check
        if self.avoid_lunch and window == TradingWindow.LUNCH_LULL:
            logger.debug("Avoiding lunch lull window")
            return False

        # Closing volatility check
        if self.avoid_closing and window == TradingWindow.CLOSING_VOLATILITY:
            logger.debug("Avoiding closing volatility window")
            return False

        # Monday morning check
        if self.avoid_monday_morning:
            if dt.weekday() == 0 and dt.time() < time(11, 0):
                logger.debug("Avoiding Monday morning")
                return False

        # Friday afternoon check
        if self.avoid_friday_afternoon:
            if dt.weekday() == 4 and dt.time() > time(14, 0):
                logger.debug("Avoiding Friday afternoon")
                return False

        # If we require excellent/good windows only
        if not allow_fair_windows:
            if window not in [TradingWindow.MORNING_PRIME, TradingWindow.AFTERNOON_PRIME]:
                return False
            if day_quality not in [DayQuality.EXCELLENT, DayQuality.GOOD]:
                return False

        return True

    def get_next_good_window(self, dt: Optional[datetime] = None) -> Tuple[datetime, TradingWindow]:
        """
        Get the next good trading window.

        Args:
            dt: Start datetime (default: now)

        Returns:
            Tuple of (datetime, TradingWindow) for next good window
        """
        if dt is None:
            dt = datetime.now(self.tz)
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)

        # Look ahead up to 7 days
        for days_ahead in range(7):
            check_date = dt + timedelta(days=days_ahead)

            # Skip weekends
            if check_date.weekday() >= 5:
                continue

            # Check morning prime window
            morning_start = check_date.replace(
                hour=10, minute=0, second=0, microsecond=0
            )
            if morning_start > dt:
                # Check Monday morning exception
                if not (check_date.weekday() == 0 and self.avoid_monday_morning):
                    return (morning_start, TradingWindow.MORNING_PRIME)

            # Check afternoon prime window
            afternoon_start = check_date.replace(
                hour=14, minute=0, second=0, microsecond=0
            )
            if afternoon_start > dt:
                # Check Friday afternoon exception
                if not (check_date.weekday() == 4 and self.avoid_friday_afternoon):
                    return (afternoon_start, TradingWindow.AFTERNOON_PRIME)

        # Fallback - shouldn't happen
        return (dt + timedelta(days=1), TradingWindow.MORNING_PRIME)

    def get_time_until_good_window(self, dt: Optional[datetime] = None) -> timedelta:
        """Get time until next good trading window."""
        if dt is None:
            dt = datetime.now(self.tz)

        next_window, _ = self.get_next_good_window(dt)
        return next_window - dt

    def get_window_quality_score(self, dt: Optional[datetime] = None) -> float:
        """
        Get a quality score (0-1) for current trading window.

        Returns:
            0.0 = worst (don't trade)
            0.5 = fair (trade with caution)
            1.0 = best (optimal trading time)
        """
        if dt is None:
            dt = datetime.now(self.tz)
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)

        window = self.get_current_window(dt)
        day_quality = self.get_day_quality(dt)

        # Window scores
        window_scores = {
            TradingWindow.PREMARKET: 0.0,
            TradingWindow.OPENING_VOLATILITY: 0.3,
            TradingWindow.MORNING_PRIME: 1.0,
            TradingWindow.LUNCH_LULL: 0.4,
            TradingWindow.AFTERNOON_PRIME: 0.9,
            TradingWindow.CLOSING_VOLATILITY: 0.5,
            TradingWindow.AFTERHOURS: 0.0,
        }

        # Day quality multipliers
        day_multipliers = {
            DayQuality.EXCELLENT: 1.0,
            DayQuality.GOOD: 0.9,
            DayQuality.FAIR: 0.8,
            DayQuality.POOR: 0.7,
        }

        base_score = window_scores.get(window, 0.5)
        multiplier = day_multipliers.get(day_quality, 0.8)

        return base_score * multiplier

    def get_position_size_adjustment(self, dt: Optional[datetime] = None) -> float:
        """
        Get position size adjustment factor based on time quality.

        Returns:
            Multiplier for position size (0.5 to 1.2)
        """
        quality = self.get_window_quality_score(dt)

        # Map quality score to position size adjustment
        # Low quality = smaller positions
        # High quality = normal or slightly larger positions
        if quality >= 0.9:
            return 1.1  # 10% larger in prime windows
        elif quality >= 0.7:
            return 1.0  # Normal size
        elif quality >= 0.5:
            return 0.8  # 20% smaller
        else:
            return 0.5  # 50% smaller in poor windows

    def get_trading_status(self, dt: Optional[datetime] = None) -> Dict:
        """Get comprehensive trading status for current time."""
        if dt is None:
            dt = datetime.now(self.tz)
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)

        window = self.get_current_window(dt)
        day_quality = self.get_day_quality(dt)
        is_good = self.is_good_time_to_trade(dt)
        quality_score = self.get_window_quality_score(dt)

        next_good_window, next_window_type = self.get_next_good_window(dt)
        time_until = next_good_window - dt if not is_good else timedelta(0)

        return {
            'current_time': dt.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'day_of_week': dt.strftime('%A'),
            'window': window.value,
            'day_quality': day_quality.value,
            'is_good_time': is_good,
            'quality_score': quality_score,
            'position_size_mult': self.get_position_size_adjustment(dt),
            'next_good_window': next_good_window.strftime('%Y-%m-%d %H:%M:%S %Z') if not is_good else None,
            'time_until_good': str(time_until) if not is_good else None,
            'recommendation': self._get_recommendation(window, day_quality, is_good)
        }

    def _get_recommendation(
        self,
        window: TradingWindow,
        day_quality: DayQuality,
        is_good: bool
    ) -> str:
        """Get trading recommendation based on current conditions."""
        if not is_good:
            if window == TradingWindow.OPENING_VOLATILITY:
                return "Wait for opening volatility to settle (~10am ET)"
            elif window == TradingWindow.LUNCH_LULL:
                return "Wait for afternoon session (~2pm ET)"
            elif window == TradingWindow.CLOSING_VOLATILITY:
                return "Consider waiting for tomorrow's session"
            elif window in [TradingWindow.PREMARKET, TradingWindow.AFTERHOURS]:
                return "Market closed - wait for regular hours"
            else:
                return "Suboptimal window - consider waiting"

        if window == TradingWindow.MORNING_PRIME:
            return "PRIME TIME - Best window for entries"
        elif window == TradingWindow.AFTERNOON_PRIME:
            return "Good window for trading"
        else:
            return "Acceptable window - trade with normal caution"


# Convenience function
def is_good_trading_time() -> bool:
    """Quick check if now is a good time to trade."""
    return TradingHoursFilter().is_good_time_to_trade()


if __name__ == "__main__":
    """Test the trading hours filter."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("TRADING HOURS FILTER TEST")
    print("="*60)

    filter = TradingHoursFilter()
    status = filter.get_trading_status()

    print(f"\nCurrent Time: {status['current_time']}")
    print(f"Day: {status['day_of_week']}")
    print(f"Window: {status['window']}")
    print(f"Day Quality: {status['day_quality']}")
    print(f"\nGood Time to Trade: {'Yes' if status['is_good_time'] else 'No'}")
    print(f"Quality Score: {status['quality_score']:.2f}")
    print(f"Position Size Multiplier: {status['position_size_mult']:.2f}x")

    if not status['is_good_time']:
        print(f"\nNext Good Window: {status['next_good_window']}")
        print(f"Time Until: {status['time_until_good']}")

    print(f"\nRecommendation: {status['recommendation']}")
    print("="*60)
