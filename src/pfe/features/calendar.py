"""Civil calendar features on real dates, independent of bar frequency or prices."""

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant
from pfe.data.sessions import _session_date, normalize_session_schedule


def calendar_index(decisions):
    if isinstance(decisions, pd.DatetimeIndex):
        if decisions.tz is None or decisions.hasnans:
            raise ValueError("decision_at requires non-missing timezone-aware timestamps")
        index = decisions.tz_convert("UTC")
    else:
        index = pd.DatetimeIndex([_instant(v, "decision_at") for v in decisions])
    if index.empty:
        index = pd.DatetimeIndex([], tz="UTC")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Decisions must be sorted unique aware timestamps")
    return index


def compute_calendar_features(
    decisions, *, timezone="America/New_York", schedule=None, session_dates=None
):
    """Civil cycles and optional exchange-session period positions.

    Daily callers pass actual economic session dates, including overnight
    sessions. Session counts require full period coverage in schedule.attrs
    start/end, or conservative bracketing by the supplied schedule range.
    No omitted holiday/session is reconstructed.
    """
    index = calendar_index(decisions)
    local = index.tz_convert(timezone)
    dates = local.tz_localize(None).normalize()
    if session_dates is not None:
        if len(session_dates) != len(index):
            raise ValueError("session_dates must align with decisions")
        dates = pd.DatetimeIndex([_session_date(v) for v in session_dates])
    iso = dates.isocalendar()
    out = {
        "calendar.year": dates.year.astype(float),
        "calendar.iso_year": iso.year.to_numpy(dtype=float),
        "calendar.month": dates.month.astype(float),
        "calendar.iso_week": iso.week.to_numpy(dtype=float),
        "calendar.quarter": dates.quarter.astype(float),
        "calendar.day_of_month": dates.day.astype(float),
        "calendar.day_of_year": dates.dayofyear.astype(float),
        "calendar.weekday": dates.dayofweek.astype(float),
        "calendar.is_weekend": (dates.dayofweek >= 5).astype(float),
        "calendar.days_in_month": dates.days_in_month.astype(float),
    }
    for period in ("month", "quarter", "year"):
        for edge in ("start", "end"):
            out[f"calendar.is_{period}_{edge}"] = getattr(dates, f"is_{period}_{edge}").astype(
                float
            )
    weeks = np.array(
        [pd.Timestamp(year=int(year), month=12, day=28).isocalendar().week for year in iso.year],
        dtype=float,
    )
    for name, values, cycle in (
        ("month", dates.month - 1, 12),
        ("weekday", dates.dayofweek, 7),
        ("iso_week", iso.week.to_numpy(dtype=float) - 1, weeks),
        ("quarter", dates.quarter - 1, 4),
        ("day_of_month", dates.day - 1, dates.days_in_month),
        ("day_of_year", dates.dayofyear - 1, np.where(dates.is_leap_year, 366, 365)),
    ):
        phase = 2 * np.pi * np.asarray(values) / cycle
        out[f"calendar.{name}_sin"] = np.sin(phase)
        out[f"calendar.{name}_cos"] = np.cos(phase)
    sessions = normalize_session_schedule(schedule) if schedule is not None else None
    if sessions is not None and "available_at" in sessions:
        sessions["available_at"] = [
            _instant(v, "schedule.available_at") for v in sessions.available_at
        ]
    coverage_start = coverage_end = None
    if sessions is not None and len(sessions):
        coverage_start = pd.Timestamp(_session_date(schedule.attrs.get("start", sessions.index[0])))
        coverage_end = pd.Timestamp(_session_date(schedule.attrs.get("end", sessions.index[-1])))
        if coverage_end < coverage_start:
            raise ValueError("Schedule coverage bounds must be ordered")
    for label, frequency in (("month", "M"), ("quarter", "Q"), ("year", "Y")):
        names = ("session_index", "sessions_remaining", "is_first_session", "is_last_session")
        for name in names:
            out[f"calendar.{name}_{label}"] = np.full(len(index), np.nan)
        out[f"calendar.mask.schedule_{label}"] = np.zeros(len(index))
        if sessions is None or sessions.empty:
            continue
        session_days = pd.DatetimeIndex(sessions.index)
        periods = dates.to_period(frequency)
        for period in periods.unique():
            if coverage_start > period.start_time or coverage_end < period.end_time.normalize():
                continue
            selected = session_days[session_days.to_period(frequency) == period]
            positions = {day: i for i, day in enumerate(selected)}
            for i in np.flatnonzero(periods == period):
                if dates[i] not in positions:
                    continue
                if "available_at" in sessions:
                    period_rows = sessions.loc[sessions.index.isin(selected.strftime("%Y-%m-%d"))]
                    if (period_rows.available_at > index[i]).any():
                        continue
                position = positions[dates[i]]
                values = (
                    position + 1,
                    len(selected) - position - 1,
                    position == 0,
                    position == len(selected) - 1,
                )
                for name, value in zip(names, values, strict=True):
                    out[f"calendar.{name}_{label}"][i] = float(value)
                out[f"calendar.mask.schedule_{label}"][i] = 1.0
    result = pd.DataFrame(out, index=index)
    result.attrs.update(
        timestamp_semantics="explicit_decision",
        calendar_timezone=timezone,
        date_basis="session_date" if session_dates is not None else "local_decision_date",
        schedule_counts="injected calendar, complete period bounds required",
    )
    return result


CALENDAR_FEATURE_NAMES = tuple(compute_calendar_features(pd.DatetimeIndex([], tz="UTC")).columns)
