import pytest

from utils.run_artifacts import JsonlWriter, write_json
from utils.run_replay import (
    filter_events,
    format_replay_report,
    load_run_artifacts,
    resolve_run_directory,
)


def _seed_run(tmp_path):
    run_id = "backtest_20240101_120000_deadbeef"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "strategy": "DummyStrategy",
            "final_equity": 101000,
            "total_return": 0.01,
        },
    )
    write_json(
        run_dir / "manifest.json",
        {"run_id": run_id, "artifacts": {"summary": "summary.json"}},
    )

    with JsonlWriter(run_dir / "decision_events.jsonl") as writer:
        writer.write(
            {
                "event_type": "decision",
                "date": "2024-01-01",
                "symbol": "AAPL",
                "action": "buy",
                "error": None,
            }
        )
        writer.write(
            {
                "event_type": "decision",
                "date": "2024-01-01",
                "symbol": "MSFT",
                "action": "sell",
                "error": "mock_failure",
            }
        )

    with JsonlWriter(run_dir / "trades.jsonl") as writer:
        writer.write(
            {
                "event_type": "trade",
                "date": "2024-01-01",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "price": 100.5,
            }
        )

    with JsonlWriter(run_dir / "order_reconciliation_events.jsonl") as writer:
        writer.write(
            {
                "event_type": "order_reconciliation_snapshot",
                "status": "ok",
                "consecutive_mismatch_runs": 0,
            }
        )

    with JsonlWriter(run_dir / "position_reconciliation_events.jsonl") as writer:
        writer.write(
            {
                "event_type": "position_reconciliation_snapshot",
                "positions_match": True,
                "mismatch_count": 0,
            }
        )

    with JsonlWriter(run_dir / "ops_slo_events.jsonl") as writer:
        writer.write(
            {
                "event_type": "slo_breach",
                "name": "data_quality_errors",
                "severity": "warning",
                "message": "example warning",
            }
        )

    with JsonlWriter(run_dir / "incident_events.jsonl") as writer:
        writer.write(
            {
                "event_type": "incident_open",
                "incident_id": "inc_test_1",
                "name": "data_quality_errors",
                "severity": "critical",
                "message": "Data quality threshold breached",
            }
        )

    with JsonlWriter(run_dir / "data_quality_events.jsonl") as writer:
        writer.write(
            {
                "event_type": "data_quality_snapshot",
                "total_errors": 0,
                "stale_warnings": 0,
            }
        )

    return run_id


def test_resolve_run_directory_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_run_directory("missing", artifacts_dir=str(tmp_path))


def test_load_run_artifacts_reads_files(tmp_path):
    run_id = _seed_run(tmp_path)
    artifacts = load_run_artifacts(run_id, artifacts_dir=str(tmp_path))
    assert artifacts["run_id"] == run_id
    assert artifacts["summary"]["strategy"] == "DummyStrategy"
    assert len(artifacts["decisions"]) == 2
    assert len(artifacts["trades"]) == 1
    assert len(artifacts["order_reconciliation"]) == 1
    assert len(artifacts["position_reconciliation"]) == 1
    assert len(artifacts["slo_events"]) == 1
    assert len(artifacts["incident_events"]) == 1
    assert len(artifacts["data_quality_events"]) == 1


def test_filter_events_symbol_date_and_errors():
    events = [
        {"symbol": "AAPL", "date": "2024-01-01", "event_type": "decision", "error": None},
        {"symbol": "MSFT", "date": "2024-01-01", "event_type": "decision", "error": "boom"},
        {"symbol": "AAPL", "date": "2024-01-02", "event_type": "trade", "error": None},
    ]

    assert len(filter_events(events, symbol="AAPL")) == 2
    assert len(filter_events(events, date_prefix="2024-01-01")) == 2
    assert len(filter_events(events, errors_only=True)) == 1
    assert len(filter_events(events, event_type="trade")) == 1


def test_format_replay_report_includes_key_fields():
    text = format_replay_report(
        summary={"run_id": "r1", "strategy": "S1", "final_equity": 100500, "total_return": 0.005},
        decisions=[{"date": "2024-01-01", "symbol": "AAPL", "action": "buy"}],
        trades=[{"date": "2024-01-01", "symbol": "AAPL", "side": "buy", "quantity": 1, "price": 100.5}],
        order_reconciliation=[{"status": "ok", "consecutive_mismatch_runs": 0}],
        position_reconciliation=[{"positions_match": True, "mismatch_count": 0}],
        slo_events=[{"severity": "warning", "name": "x", "message": "y"}],
        incident_events=[{"event_type": "incident_open", "incident_id": "inc_1"}],
        data_quality_events=[{"total_errors": 0, "stale_warnings": 0}],
        limit=10,
    )
    assert "REPLAY REPORT" in text
    assert "run_id=r1" in text
    assert "AAPL" in text
    assert "Operations Timeline" in text
    assert "Latest Incident" in text
