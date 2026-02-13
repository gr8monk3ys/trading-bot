"""
Data quality validation utilities for market and PIT datasets.

These checks are lightweight and intended for pre-trade/backtest guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, List, Optional

import pandas as pd


@dataclass
class DataQualityIssue:
    """Represents one quality issue found during validation."""

    severity: str  # "error" or "warning"
    code: str
    message: str
    symbol: str = ""
    affected_rows: int = 0


@dataclass
class DataQualityReport:
    """Validation report for a single symbol/dataset."""

    symbol: str
    rows: int
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    issues: List[DataQualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rows": self.rows,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "has_errors": self.has_errors,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "symbol": i.symbol,
                    "affected_rows": i.affected_rows,
                }
                for i in self.issues
            ],
        }


def validate_ohlcv_frame(
    df: pd.DataFrame,
    symbol: str = "",
    stale_after_days: Optional[int] = None,
    reference_time: Optional[datetime] = None,
) -> DataQualityReport:
    """
    Validate a standard OHLCV DataFrame.

    Required columns: `open`, `high`, `low`, `close`, `volume`.
    """
    report = DataQualityReport(
        symbol=symbol,
        rows=len(df) if isinstance(df, pd.DataFrame) else 0,
    )

    if not isinstance(df, pd.DataFrame):
        report.issues.append(
            DataQualityIssue(
                severity="error",
                code="invalid_type",
                message="Dataset is not a pandas DataFrame",
                symbol=symbol,
            )
        )
        return report

    if len(df) == 0:
        report.issues.append(
            DataQualityIssue(
                severity="error",
                code="empty_frame",
                message="Dataset is empty",
                symbol=symbol,
            )
        )
        return report

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report.issues.append(
            DataQualityIssue(
                severity="error",
                code="missing_columns",
                message=f"Missing required columns: {', '.join(missing_cols)}",
                symbol=symbol,
            )
        )
        return report

    if isinstance(df.index, pd.DatetimeIndex):
        report.start = (
            df.index.min().to_pydatetime()
            if hasattr(df.index.min(), "to_pydatetime")
            else df.index.min()
        )
        report.end = (
            df.index.max().to_pydatetime()
            if hasattr(df.index.max(), "to_pydatetime")
            else df.index.max()
        )
        if not df.index.is_monotonic_increasing:
            report.issues.append(
                DataQualityIssue(
                    severity="error",
                    code="non_monotonic_index",
                    message="Datetime index is not monotonic increasing",
                    symbol=symbol,
                )
            )
        duplicate_count = int(df.index.duplicated().sum())
        if duplicate_count > 0:
            report.issues.append(
                DataQualityIssue(
                    severity="warning",
                    code="duplicate_timestamps",
                    message="Duplicate timestamps detected",
                    symbol=symbol,
                    affected_rows=duplicate_count,
                )
            )
    else:
        report.issues.append(
            DataQualityIssue(
                severity="warning",
                code="non_datetime_index",
                message="Index is not a DatetimeIndex",
                symbol=symbol,
            )
        )

    null_rows = int(df[required_cols].isnull().any(axis=1).sum())
    if null_rows > 0:
        report.issues.append(
            DataQualityIssue(
                severity="warning",
                code="null_values",
                message="Rows with null OHLCV values detected",
                symbol=symbol,
                affected_rows=null_rows,
            )
        )

    non_positive_price_rows = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if non_positive_price_rows > 0:
        report.issues.append(
            DataQualityIssue(
                severity="error",
                code="non_positive_prices",
                message="Non-positive prices detected",
                symbol=symbol,
                affected_rows=non_positive_price_rows,
            )
        )

    negative_volume_rows = int((df["volume"] < 0).sum())
    if negative_volume_rows > 0:
        report.issues.append(
            DataQualityIssue(
                severity="error",
                code="negative_volume",
                message="Negative volume values detected",
                symbol=symbol,
                affected_rows=negative_volume_rows,
            )
        )

    invalid_high_low_mask = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    invalid_high_low_rows = int(invalid_high_low_mask.sum())
    if invalid_high_low_rows > 0:
        report.issues.append(
            DataQualityIssue(
                severity="error",
                code="invalid_ohlc_bounds",
                message="OHLC bounds violation (high/low inconsistent)",
                symbol=symbol,
                affected_rows=invalid_high_low_rows,
            )
        )

    if stale_after_days is not None and report.end is not None:
        ref = reference_time or datetime.now()
        age_days = (ref - report.end).days
        if age_days > stale_after_days:
            report.issues.append(
                DataQualityIssue(
                    severity="warning",
                    code="stale_data",
                    message=f"Latest bar is stale by {age_days} days",
                    symbol=symbol,
                )
            )

    return report


def validate_pit_announcement_dates(
    data_points: Iterable[Any],
    as_of_date: datetime,
    symbol: str = "",
) -> DataQualityReport:
    """
    Validate point-in-time announcement discipline.

    Flags data points whose announcement timestamp is after the query as-of date.
    """
    points = list(data_points)
    report = DataQualityReport(symbol=symbol, rows=len(points))

    lookahead_count = 0
    missing_announcement_count = 0

    for point in points:
        announced = getattr(point, "announced_date", None)
        if announced is None and isinstance(point, dict):
            announced = point.get("announced_date")
        if announced is None:
            missing_announcement_count += 1
            continue
        if isinstance(announced, str):
            try:
                announced = datetime.fromisoformat(announced)
            except ValueError:
                missing_announcement_count += 1
                continue
        if announced > as_of_date:
            lookahead_count += 1

    if lookahead_count > 0:
        report.issues.append(
            DataQualityIssue(
                severity="error",
                code="pit_lookahead_violation",
                message="Data points include values announced after as-of date",
                symbol=symbol,
                affected_rows=lookahead_count,
            )
        )

    if missing_announcement_count > 0:
        report.issues.append(
            DataQualityIssue(
                severity="warning",
                code="missing_announcement_date",
                message="Data points missing parseable announcement dates",
                symbol=symbol,
                affected_rows=missing_announcement_count,
            )
        )

    return report


def summarize_quality_reports(
    reports: Iterable[DataQualityReport | dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate data quality reports for operational gating.
    """
    normalized: list[dict[str, Any]] = []

    for report in reports:
        if isinstance(report, DataQualityReport):
            normalized.append(report.to_dict())
        elif isinstance(report, dict):
            normalized.append(report)

    total_errors = 0
    total_warnings = 0
    stale_warnings = 0
    symbols_with_errors = 0

    for report in normalized:
        error_count = int(report.get("error_count", 0) or 0)
        warning_count = int(report.get("warning_count", 0) or 0)
        issues = report.get("issues", []) or []

        total_errors += error_count
        total_warnings += warning_count
        if error_count > 0:
            symbols_with_errors += 1

        stale_warnings += sum(
            1
            for issue in issues
            if isinstance(issue, dict) and issue.get("code") == "stale_data"
        )

    return {
        "reports": normalized,
        "symbols_checked": len(normalized),
        "symbols_with_errors": symbols_with_errors,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "stale_warnings": stale_warnings,
    }


def should_halt_trading_for_data_quality(
    summary: dict[str, Any],
    *,
    max_errors: int = 0,
    max_stale_warnings: int = 0,
) -> tuple[bool, Optional[str]]:
    """
    Decide if trading entries should halt based on aggregate data quality.
    """
    errors = int(summary.get("total_errors", 0) or 0)
    stale_warnings = int(summary.get("stale_warnings", 0) or 0)

    if errors > max_errors:
        return True, f"Data quality errors {errors} exceeded max {max_errors}"

    if stale_warnings > max_stale_warnings:
        return (
            True,
            f"Stale data warnings {stale_warnings} exceeded max {max_stale_warnings}",
        )

    return False, None
