from __future__ import annotations

import json

from utils.secrets_audit import run_secrets_audit, sanitize_audit_report


def test_secrets_audit_passes_with_fresh_inventory_and_no_hits(tmp_path):
    repo_root = tmp_path / "repo"
    docs = repo_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (repo_root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (docs / "SECRETS_ROTATION_INVENTORY.json").write_text(
        json.dumps(
            {
                "secrets": [
                    {"name": "ALPACA_API_KEY", "last_rotated": "2026-01-15", "max_age_days": 365}
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_secrets_audit(repo_root=repo_root)

    assert report["ready"] is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["secrets_inventory_present"]["passed"] is True
    assert checks["repository_secret_leak_scan"]["passed"] is True


def test_secrets_audit_detects_probable_inline_secret(tmp_path):
    repo_root = tmp_path / "repo"
    docs = repo_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (repo_root / "config.py").write_text(
        'ALPACA_SECRET_KEY = "abcDEF1234567890TOKENVALUE"\n',
        encoding="utf-8",
    )
    (docs / "SECRETS_ROTATION_INVENTORY.json").write_text(
        json.dumps({"secrets": [{"name": "ALPACA_SECRET_KEY", "last_rotated": "2026-01-15"}]}),
        encoding="utf-8",
    )

    report = run_secrets_audit(repo_root=repo_root)

    assert report["ready"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["repository_secret_leak_scan"]["passed"] is False


def test_secrets_audit_redacts_secret_details(tmp_path):
    repo_root = tmp_path / "repo"
    docs = repo_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    secret_value = "abcDEF1234567890TOKENVALUE"
    (repo_root / "config.py").write_text(
        f'ALPACA_SECRET_KEY = "{secret_value}"\n',
        encoding="utf-8",
    )
    (docs / "SECRETS_ROTATION_INVENTORY.json").write_text(
        json.dumps({"secrets": [{"name": "ALPACA_SECRET_KEY", "last_rotated": "2024-01-15"}]}),
        encoding="utf-8",
    )

    report = run_secrets_audit(repo_root=repo_root)
    safe_report = sanitize_audit_report(report)
    checks = {check["name"]: check for check in safe_report["checks"]}

    stale_entry = checks["secrets_rotation_fresh"]["details"]["stale_entries"][0]
    leak_hit = checks["repository_secret_leak_scan"]["details"]["hits"][0]

    assert stale_entry["name"] != "ALPACA_SECRET_KEY"
    assert leak_hit["file"] == "config.py"
    assert leak_hit["line"] == 1
    assert "excerpt" not in leak_hit


def test_secrets_audit_sanitizes_secret_values_in_report(tmp_path):
    repo_root = tmp_path / "repo"
    docs = repo_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    secret_line = 'ALPACA_SECRET_KEY = "abcDEF1234567890TOKENVALUE"\n'
    (repo_root / "config.py").write_text(secret_line, encoding="utf-8")
    (docs / "SECRETS_ROTATION_INVENTORY.json").write_text(
        json.dumps({"secrets": [{"name": "ALPACA_SECRET_KEY", "last_rotated": "2026-01-15"}]}),
        encoding="utf-8",
    )

    report = run_secrets_audit(repo_root=repo_root)
    safe_report = sanitize_audit_report(report)
    serialized = json.dumps(safe_report)

    assert "abcDEF1234567890TOKENVALUE" not in serialized
    assert "config.py" in serialized
