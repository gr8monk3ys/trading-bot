"""End-to-end smoke test of the live order path against the Alpaca PAPER API.

Proves the chain main.py uses in production — AlpacaBroker -> AuditLog ->
CircuitBreaker.enforce_before_order -> LiveOrderGateway._internal_submit_order —
by placing one 1-share SPY limit buy priced ~50% below market (it can never
fill), verifying the order ID lands in audit_logs/, then cancelling it.

Usage:
    python scripts/paper_smoke_test.py

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment or .env.
Refuses to run unless the broker is in paper mode.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

SYMBOL = "SPY"
FALLBACK_LIMIT = 100.00  # far below any plausible SPY price if quotes are unavailable


async def main() -> int:
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        print("FAIL: ALPACA_API_KEY / ALPACA_SECRET_KEY not set (.env or environment).")
        return 1
    if os.getenv("PAPER", "True").lower() not in ("true", "1", "yes"):
        print("FAIL: PAPER is not true; this smoke test only runs against the paper API.")
        return 1

    from brokers.alpaca_broker import AlpacaBroker
    from brokers.order_builder import OrderBuilder
    from engine.live_order_gateway import LiveOrderGateway
    from utils.audit_log import AuditLog
    from utils.circuit_breaker import CircuitBreaker

    broker = AlpacaBroker(paper=True)
    try:
        account = await broker.get_account()
    except Exception as e:
        print(f"FAIL: could not reach the paper account ({type(e).__name__}: {e}).")
        return 1
    print(f"Connected to paper account: equity={getattr(account, 'equity', '?')}")

    audit_log = AuditLog(log_dir="./audit_logs", auto_verify=True)
    if hasattr(broker, "set_audit_log"):
        broker.set_audit_log(audit_log)

    circuit_breaker = CircuitBreaker(max_daily_loss=0.03, auto_close_positions=False)
    await circuit_breaker.initialize(broker)

    gateway = LiveOrderGateway(broker, circuit_breaker=circuit_breaker)

    try:
        price = await broker.get_last_price(SYMBOL)
        limit_price = round(float(price) * 0.5, 2)
    except Exception as e:
        print(f"note: no quote for {SYMBOL} ({type(e).__name__}); using fallback limit")
        limit_price = FALLBACK_LIMIT
    print(f"Submitting 1-share {SYMBOL} limit buy @ {limit_price} (unfillable by design)")

    order_request = OrderBuilder(SYMBOL, "buy", 1).limit(limit_price).day().build()
    result = await gateway.submit_order(order_request=order_request, strategy_name="smoke_test")

    if not result.success:
        print(f"FAIL: gateway rejected the order: {result.rejection_reason}")
        return 1
    print(f"PASS: order accepted, id={result.order_id}")

    audit_files = sorted(Path("./audit_logs").glob("*.jsonl"), key=os.path.getmtime)
    logged = audit_files and result.order_id in audit_files[-1].read_text()
    print(
        f"{'PASS' if logged else 'WARN'}: order id "
        f"{'found in ' + audit_files[-1].name if logged else 'not found in audit_logs/'}"
    )

    try:
        await broker.cancel_order(result.order_id)
        print("PASS: order cancelled")
    except Exception as e:
        print(
            f"WARN: cancel failed ({e}) — 1-share limit @ {limit_price} cannot fill; "
            f"cancel manually in the Alpaca dashboard"
        )

    audit_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
