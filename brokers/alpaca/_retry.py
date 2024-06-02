"""
Shared utilities for the AlpacaBroker submodules.

Contains:
    - retry_with_backoff: async retry decorator with exponential backoff and jitter
    - Custom exception classes (BrokerError hierarchy + GatewayBypassError)
    - DEBUG_MODE flag and module logger

Kept separate from the sub-client mixins so each module can import from a
common, low-dependency location without circularity.
"""

import asyncio
import logging
import os
import random
from functools import wraps

# P2 FIX: Environment-aware logging - only show full tracebacks in debug mode
# This prevents sensitive information from leaking in production logs
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")

logger = logging.getLogger(__name__)


# P2 FIX: Custom exception for broker errors
class BrokerError(Exception):
    """
    Exception raised for broker operation failures.

    Use this for critical errors where the caller MUST handle the failure
    (e.g., order submission failures, authentication errors).
    """

    pass


class BrokerConnectionError(BrokerError):
    """Raised when broker connection fails."""

    pass


class OrderError(BrokerError):
    """Raised when order operations fail."""

    pass


class GatewayBypassError(BrokerError):
    """
    Raised when attempting to submit orders without using OrderGateway.

    CRITICAL SAFETY: All orders MUST route through OrderGateway to ensure:
    - Circuit breaker checks
    - Position conflict detection
    - Risk manager limits enforcement
    - Audit trail maintenance

    To fix this error, use order_gateway.submit_order() instead of
    broker.submit_order_advanced() directly.
    """

    pass


# ERROR HANDLING CONVENTIONS:
# This module uses the following patterns for error handling:
#
# 1. QUERY methods (get_position, get_last_price, get_bars):
#    - Return None or [] on error
#    - Log the error
#    - Caller should check for None/empty before using
#
# 2. ACTION methods (submit_order, cancel_order):
#    - Raise OrderError on critical failures
#    - Return False for non-critical failures (already cancelled, etc.)
#    - Log the error
#
# 3. CONNECTION methods (get_account, get_positions):
#    - Raise BrokerConnectionError if broker is unreachable
#    - These are critical - trading cannot proceed without them


def retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10, jitter=0.1):
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        jitter: Jitter factor (0.0-1.0) to randomize delay and prevent thundering herd

    The jitter adds randomness to prevent many clients from retrying simultaneously.
    For example, with jitter=0.1 and base delay of 2s, actual delay will be 1.8s-2.2s.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError, OSError) as e:
                    # Network-related errors are retryable
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed (network error): {e}"
                    )
                except Exception as e:
                    # For other exceptions, check if they seem transient
                    error_str = str(e).lower()
                    is_transient = any(
                        term in error_str
                        for term in [
                            "timeout",
                            "connection",
                            "rate limit",
                            "429",
                            "503",
                            "502",
                            "504",
                        ]
                    )

                    if is_transient:
                        last_exception = e
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed (transient): {e}"
                        )
                    else:
                        # Non-transient error, don't retry
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                if attempt < max_retries - 1:
                    # Calculate base delay with exponential backoff
                    base_delay = min(initial_delay * (2**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    # Jitter range: [base * (1 - jitter), base * (1 + jitter)]
                    jitter_range = base_delay * jitter
                    sleep_time = base_delay + random.uniform(-jitter_range, jitter_range)
                    sleep_time = max(0.1, sleep_time)  # Ensure minimum delay

                    logger.info(f"Retrying {func.__name__} in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)

            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper

    return decorator
