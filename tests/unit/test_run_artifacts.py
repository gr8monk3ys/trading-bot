from datetime import datetime

from utils.run_artifacts import (
    JsonlWriter,
    ensure_run_directory,
    generate_run_id,
    read_json,
    read_jsonl,
    write_json,
)


def test_generate_run_id_has_prefix_and_timestamp():
    run_id = generate_run_id(prefix="backtest", now=datetime(2024, 1, 2, 3, 4, 5))
    assert run_id.startswith("backtest_20240102_030405_")
    assert len(run_id.split("_")[-1]) == 8


def test_ensure_run_directory_creates_path(tmp_path):
    run_dir = ensure_run_directory(tmp_path, "demo_run")
    assert run_dir.exists()
    assert run_dir.is_dir()


def test_json_roundtrip(tmp_path):
    out = tmp_path / "summary.json"
    write_json(
        out,
        {
            "run_id": "demo",
            "completed_at": datetime(2024, 1, 1, 9, 30, 0),
            "value": 123,
        },
    )
    loaded = read_json(out)
    assert loaded["run_id"] == "demo"
    assert loaded["completed_at"].startswith("2024-01-01T09:30:00")
    assert loaded["value"] == 123


def test_jsonl_writer_roundtrip(tmp_path):
    out = tmp_path / "events.jsonl"
    with JsonlWriter(out) as writer:
        writer.write(
            {
                "event_type": "decision",
                "date": "2024-01-01",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0),
            }
        )
        writer.write({"event_type": "trade", "date": "2024-01-02"})

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["event_type"] == "decision"
    assert rows[0]["timestamp"].startswith("2024-01-01T10:00:00")
    assert rows[1]["event_type"] == "trade"
