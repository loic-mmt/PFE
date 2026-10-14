"""Exchange sessions with actual opens, closes and optional trading breaks."""

import re
from datetime import date, datetime

import pandas as pd

from pfe.data.contracts import _frame, _timestamps


def _session_date(value):
    """Read an explicitly date-labelled session, never infer a trading instant."""
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as error:
            raise ValueError("session_date requires a valid calendar date") from error
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp, datetime, str)):
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError) as error:
            raise ValueError("session_date requires an ISO date or midnight date label") from error
        if not pd.isna(stamp) and stamp == stamp.normalize():
            return stamp.date().isoformat()
    raise ValueError("session_date requires an ISO date or midnight date label")


def normalize_session_schedule(frame):
    """Validate an injected exchange schedule, without inventing missing sessions.

    Requires open_at/close_at and either a session_date column or date-labelled
    index. Returns unique ISO session dates as index, actual aware UTC instants
    as columns. Economic date labels may be encoded at midnight but never become
    open/close times. Sessions cannot overlap. Optional break_start_at and
    break_end_at must both exist and lie strictly inside the session when present.
    Calendar accuracy is the caller's responsibility; validation cannot prove
    omitted holidays or detect an omitted expected trading session.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Schedule requires a DataFrame")
    labels = frame["session_date"] if "session_date" in frame else frame.index
    out = _frame(frame, ("open_at", "close_at"), "Schedule")
    out["session_date"] = [_session_date(value) for value in labels]
    if out.session_date.duplicated().any():
        raise ValueError("Duplicate session_date")
    for name in ("open_at", "close_at"):
        out[name] = _timestamps(out[name], name)
    if (out.close_at <= out.open_at).any():
        raise ValueError("close_at must be later than open_at")
    breaks = ("break_start_at", "break_end_at")
    if any(name in out for name in breaks) and not all(name in out for name in breaks):
        raise ValueError("Trading breaks require break_start_at and break_end_at")
    for name in breaks:
        if name not in out:
            out[name] = pd.NaT
        out[name] = _timestamps(out[name], name, nullable=True)
    if out.break_start_at.isna().ne(out.break_end_at.isna()).any():
        raise ValueError("Trading break bounds must both be present or absent")
    present = out.break_start_at.notna()
    if (
        (out.loc[present, "break_start_at"] <= out.loc[present, "open_at"])
        | (out.loc[present, "break_end_at"] <= out.loc[present, "break_start_at"])
        | (out.loc[present, "break_end_at"] >= out.loc[present, "close_at"])
    ).any():
        raise ValueError("Trading breaks must lie strictly inside their session")
    out = out.sort_values("session_date", kind="stable", ignore_index=True)
    if not out.open_at.is_monotonic_increasing:
        raise ValueError("Session dates and opens must have the same chronological order")
    if (
        out.open_at.iloc[1:].reset_index(drop=True) < out.close_at.iloc[:-1].reset_index(drop=True)
    ).any():
        raise ValueError("Exchange sessions cannot overlap")
    return out.set_index("session_date")


def exchange_schedule(start, end, calendar="XNYS"):
    """Return inclusive session-date bounds from an exchange calendar.

    Named calendars use the optional ``calendars`` extra (exchange_calendars),
    which calculates holidays, DST and early closes locally. No network call.
    A DataFrame is an injectable schedule; calendar objects exposing .schedule
    are also accepted. Injected schedules are validated but are not independently
    certified as complete exchange history. start/end are economic session dates.
    """
    first, last = _session_date(start), _session_date(end)
    if last < first:
        raise ValueError("start must not follow end")
    if isinstance(calendar, str):
        if not calendar or calendar != calendar.strip():
            raise ValueError("calendar requires a non-empty exchange calendar name")
        try:
            import exchange_calendars as calendars
        except ImportError as error:
            raise ImportError(
                "Named exchange calendars require the pfe[calendars] extra"
            ) from error
        # Constructors require at least one session and start < end. Padding
        # permits a holiday-only/weekend-only request to return a valid empty
        # schedule after slicing, rather than failing calendar construction.
        try:
            calendar = calendars.get_calendar(
                calendar,
                start=pd.Timestamp(first) - pd.Timedelta("14D"),
                end=pd.Timestamp(last) + pd.Timedelta("14D"),
            )
        except calendars.errors.CalendarError as error:
            raise ValueError(f"Cannot create exchange calendar {calendar!r}: {error}") from error
    if isinstance(calendar, pd.DataFrame):
        schedule = calendar
        name, timezone = "injected", calendar.attrs.get("timezone")
    elif hasattr(calendar, "schedule"):
        schedule = calendar.schedule
        name, timezone = getattr(calendar, "name", "injected"), str(getattr(calendar, "tz", ""))
    else:
        raise ValueError("calendar requires a name, schedule DataFrame or calendar object")
    schedule = schedule.rename(
        columns={
            "open": "open_at",
            "market_open": "open_at",
            "close": "close_at",
            "market_close": "close_at",
            "break_start": "break_start_at",
            "break_end": "break_end_at",
        }
    )
    out = normalize_session_schedule(schedule)
    out = out.loc[(out.index >= first) & (out.index <= last)].copy()
    out.attrs.update(calendar=name, timezone=timezone, start=first, end=last)
    return out
