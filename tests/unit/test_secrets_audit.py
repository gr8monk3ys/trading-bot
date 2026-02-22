from __future__ import annotations

import json

from utils.secrets_audit import run_secrets_audit


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
