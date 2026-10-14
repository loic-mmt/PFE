"""Macro observations and announcements with explicit availability timestamps."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_COLUMNS = ("series", "observation_at", "available_at", "value", "point_in_time")
CALENDAR_COLUMNS = (
    "event_id",
    "event",
    "scheduled_at",
    "known_at",
    "actual",
    "forecast",
    "previous",
    "actual_available_at",
    "forecast_available_at",
    "previous_available_at",
    "point_in_time",
)


def _timestamps(values, name):
    result = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tz is None:
            raise ValueError(f"{name} requires non-missing timezone-aware timestamps")
        result.append(stamp.tz_convert("UTC"))
    return pd.DatetimeIndex(result, tz="UTC")


def _quality(frame):
    if "point_in_time" not in frame:
        frame["point_in_time"] = True
    if not frame.point_in_time.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("point_in_time must contain booleans")


def _numbers(values, name, *, nullable=False):
    if values.map(lambda x: isinstance(x, (bool, np.bool_, complex))).any():
        raise ValueError(f"{name} must contain real numbers")
    result = pd.to_numeric(values, errors="raise").astype(float)
    if np.isinf(result).any() or (not nullable and result.isna().any()):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def normalize_macro_series(frame):
    """Validate long-form observations; preserve vintages, including older-period revisions.

    Required columns: series, observation_at, available_at, value. Naive dates are
    rejected: an observation period must never silently become its publication date.
    ``point_in_time`` defaults to True for caller-supplied, explicit vintage data.
    """
    required = set(SERIES_COLUMNS[:-1])
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError(f"Macro series require {sorted(required)}")
    out = frame.copy()
    if not out.series.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("series must contain non-empty strings")
    for name in ("observation_at", "available_at"):
        out[name] = _timestamps(out[name], name)
    out["value"] = _numbers(out.value, "value", nullable=True)
    _quality(out)
    if out.duplicated(["series", "observation_at", "available_at"]).any():
        raise ValueError("Duplicate observation vintage")
    return out.loc[:, SERIES_COLUMNS].sort_values(
        ["series", "available_at", "observation_at"], kind="stable", ignore_index=True
    )


def normalize_macro_calendar(frame):
    """Validate one row per event, with separate schedule and field availability.

    Required: event_id, event, scheduled_at, known_at. Optional numeric fields actual,
    forecast, previous each require their own *_available_at when present. Historical
    schedule revisions must be resolved into a point-in-time input before this API.
    """
    required = set(CALENDAR_COLUMNS[:4])
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError(f"Macro calendar requires {sorted(required)}")
    out = frame.copy()
    if out.event_id.isna().any() or out.event_id.duplicated().any():
        raise ValueError("event_id must be non-missing and unique")
    if not out.event.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("event must contain non-empty strings")
    for name in ("scheduled_at", "known_at"):
        out[name] = _timestamps(out[name], name)
    for name in ("actual", "forecast", "previous"):
        if name not in out:
            out[name] = np.nan
        out[name] = _numbers(out[name], name, nullable=True)
        available = f"{name}_available_at"
        if available not in out:
            out[available] = pd.NaT
        present = out[name].notna()
        stamps = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        stamps.loc[present] = _timestamps(out.loc[present, available], available)
        out[available] = stamps
        if (out.loc[present, available] < out.loc[present, "known_at"]).any():
            raise ValueError(f"{available} cannot precede known_at")
    _quality(out)
    return out.loc[:, CALENDAR_COLUMNS].sort_values("scheduled_at", ignore_index=True)


def _lse_number(value):
    """Provider percentages retain percentage-point units; K/M/B are multipliers."""
    if value is None or str(value).strip() in ("", "-", "--", "N/A", "n/a"):
        return np.nan
    if isinstance(value, (bool, complex)):
        raise ValueError(f"Unsupported LSE numeric value: {value!r}")
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([%KMB]?)",
        str(value).strip().replace(",", ""),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unsupported LSE numeric value: {value!r}")
    return float(match[1]) * {"": 1, "%": 1, "K": 1e3, "M": 1e6, "B": 1e9}[match[2].upper()]


@dataclass(frozen=True)
class MacroData:
    """Normalized data from a bounded latest-value LSE snapshot, not historical vintages."""

    series: pd.DataFrame
    calendar: pd.DataFrame
    snapshot_at: pd.Timestamp
    limit_reached: dict


def load_macro(path, *, max_bytes=1048576):
    """Read downloader JSON. All latest-snapshot values become available at receipt.

    LSE series dates are observation periods. Calendar datetime is a scheduled time;
    created_at/updated_at are ingestion metadata. None establish historical vintage
    availability, so this loader deliberately does not backdate any values.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    with Path(path).open("rb") as file:
        raw = file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Macro file exceeds max_bytes")
    payload = json.loads(raw)
    if payload.get("provider") != "LSE":
        raise ValueError("Expected an LSE macro snapshot")
    snapshot = _timestamps([payload["snapshot_at"]], "snapshot_at")[0]
    series = []
    for symbol, rows in payload["series"].items():
        for row in rows:
            series.append(
                dict(
                    series=symbol,
                    observation_at=pd.to_datetime(row["date"], utc=True),
                    available_at=snapshot,
                    value=_lse_number(row["value"]),
                    point_in_time=False,
                )
            )
    calendar = []
    for row in payload["calendar"]:
        values = {key: _lse_number(row.get(key)) for key in ("actual", "previous")}
        # Consensus is the expectation benchmark; provider forecast is a fallback.
        forecast = _lse_number(row.get("consensus"))
        values["forecast"] = _lse_number(row.get("forecast")) if np.isnan(forecast) else forecast
        scheduled = _timestamps([row["datetime"]], "datetime")[0]
        calendar.append(
            dict(
                event_id=row["id"],
                event=row["event"],
                scheduled_at=scheduled,
                known_at=snapshot,
                point_in_time=False,
                **values,
                **{f"{key}_available_at": snapshot for key in values},
            )
        )
    return MacroData(
        normalize_macro_series(pd.DataFrame(series, columns=SERIES_COLUMNS)),
        normalize_macro_calendar(pd.DataFrame(calendar, columns=CALENDAR_COLUMNS)),
        snapshot,
        payload.get(
            "limit_reached",
            {
                **{
                    symbol: len(rows) == payload.get("limit")
                    for symbol, rows in payload["series"].items()
                },
                "calendar": len(payload["calendar"]) == payload.get("limit"),
            },
        ),
    )
