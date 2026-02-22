from __future__ import annotations

from incident_contacts import validate_incident_contacts


def test_validate_incident_contacts_reports_placeholders(tmp_path):
    ownership = tmp_path / "ownership.md"
    escalation = tmp_path / "escalation.md"

    ownership.write_text(
        "\n".join(
            [
                "# Ownership",
                "- Service owner: `REPLACE_WITH_OWNER_TEAM`",
            ]
        ),
        encoding="utf-8",
    )
    escalation.write_text(
        "\n".join(
            [
                "# Escalation",
                "- Pager policy URL: `REPLACE_WITH_PAGER_POLICY_URL`",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_incident_contacts(ownership, escalation)
    assert report["valid"] is False
    assert report["placeholder_count"] == 2
    assert len(report["findings"]) == 2


def test_validate_incident_contacts_passes_without_placeholders(tmp_path):
    ownership = tmp_path / "ownership.md"
    escalation = tmp_path / "escalation.md"

    ownership.write_text(
        "\n".join(
            [
                "# Ownership",
                "- Service owner: `Trading Systems`",
                "- Last reviewed: `2026-02-21`",
            ]
        ),
        encoding="utf-8",
    )
    escalation.write_text(
        "\n".join(
            [
                "# Escalation",
                "- Pager policy URL: `https://pager.example.internal/policy/trading`",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_incident_contacts(ownership, escalation)
    assert report["valid"] is True
    assert report["placeholder_count"] == 0
    assert report["findings"] == []
