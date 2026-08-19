"""Audit-trail coverage for the live order path.

The hash-chained audit log is the only durable record of what the bot did
with the account. `brokers/alpaca/orders.py` previously logged only
ORDER_MODIFIED (the replace path): a live smoke test on 2026-08-18 placed and
cancelled a real paper order while the day's audit file stayed at 0 bytes.
These tests pin the fix: submissions, rejections, and cancels must each land
in the attached audit log.
"""

import json
from unittest.mock import Mock, patch

from utils.audit_log import AuditLog


def _read_events(log_dir):
    events = []
    for path in sorted(log_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def _make_broker(mock_trading, tmp_path):
    from brokers.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(paper=True)
    audit = AuditLog(log_dir=str(tmp_path))
    broker.set_audit_log(audit)
    return broker, audit


def _order_request(symbol="SPY", qty="1", side="buy"):
    req = Mock()
    req.symbol = symbol
    req.qty = qty
    req.side = side
    return req


def _order_result(order_id="order-abc", symbol="SPY", qty="1"):
    result = Mock()
    result.id = order_id
    result.symbol = symbol
    result.qty = qty
    result.side = "buy"
    result.notional = None
    result.type = "limit"
    result.order_class = "simple"
    return result


@patch("brokers.alpaca_broker.StockDataStream")
@patch("brokers.alpaca_broker.StockHistoricalDataClient")
@patch("brokers.alpaca_broker.TradingClient")
async def test_internal_submit_writes_order_submitted(
    mock_trading, mock_data, mock_stream, tmp_path
):
    mock_trading.return_value.submit_order.return_value = _order_result(order_id="order-abc")
    broker, audit = _make_broker(mock_trading, tmp_path)

    await broker._internal_submit_order(_order_request(), gateway_token=None, check_impact=False)
    audit.close()

    submitted = [e for e in _read_events(tmp_path) if e["event_type"] == "order_submitted"]
    assert len(submitted) == 1
    assert submitted[0]["data"]["order_id"] == "order-abc"
    assert submitted[0]["data"]["symbol"] == "SPY"


@patch("brokers.alpaca_broker.StockDataStream")
@patch("brokers.alpaca_broker.StockHistoricalDataClient")
@patch("brokers.alpaca_broker.TradingClient")
async def test_internal_submit_failure_writes_order_rejected(
    mock_trading, mock_data, mock_stream, tmp_path
):
    mock_trading.return_value.submit_order.side_effect = RuntimeError("insufficient buying power")
    broker, audit = _make_broker(mock_trading, tmp_path)

    try:
        await broker._internal_submit_order(
            _order_request(), gateway_token=None, check_impact=False
        )
        raised = False
    except RuntimeError:
        raised = True
    audit.close()

    assert raised, "submission failure must still propagate to the caller"
    rejected = [e for e in _read_events(tmp_path) if e["event_type"] == "order_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["data"]["symbol"] == "SPY"
    assert "insufficient buying power" in rejected[0]["data"]["error"]


@patch("brokers.alpaca_broker.StockDataStream")
@patch("brokers.alpaca_broker.StockHistoricalDataClient")
@patch("brokers.alpaca_broker.TradingClient")
async def test_cancel_order_writes_order_canceled(mock_trading, mock_data, mock_stream, tmp_path):
    mock_trading.return_value.cancel_order_by_id.return_value = None
    broker, audit = _make_broker(mock_trading, tmp_path)

    assert await broker.cancel_order("order-abc") is True
    audit.close()

    canceled = [e for e in _read_events(tmp_path) if e["event_type"] == "order_canceled"]
    assert len(canceled) == 1
    assert canceled[0]["data"]["order_id"] == "order-abc"
