"""Daily bars labelled by actual exchange sessions, with explicit provenance."""

from pathlib import Path

import numpy as np
import pandas as pd

from pfe.data.bars import OHLCV, normalize_bars
from pfe.data.contracts import _frame, _identifiers, _missing, _provenance, _timestamps
from pfe.data.sessions import _session_date, normalize_session_schedule

PRICE_ADJUSTMENTS = ("raw", "split_adjusted", "total_return", "unknown")
VOLUME_ADJUSTMENTS = ("raw", "split_adjusted", "unknown")
SESSION_SCOPES = ("rth", "extended", "all", "unknown")


def load_daily_bars(path, *, schedule, **normalization_options):
    """Read canonical daily CSV/Parquet while retaining identities and provenance.

    CSV IDs are loaded as strings, preserving leading zeroes. Canonical CSV
    booleans (True/False, case-insensitive) are decoded explicitly; other strings
    remain invalid. Missing values keep their masks. Source date labels and real
    aware timestamps are validated by normalize_daily_bars rather than guessed
    by the file reader. No price adjustment or calendar inference is performed.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        identifiers = (
            "instrument_id",
            "entity_id",
            "source_id",
            "revision_id",
            "schema_version",
            "currency",
            "ticker",
            "source_document_id",
        )
        frame = pd.read_csv(path, dtype=dict.fromkeys(identifiers, str))
        for name in frame:
            if name == "point_in_time" or name.startswith("quality."):
                if name != "point_in_time" and name not in (
                    "quality.session_complete",
                    "quality.coverage_verified",
                    "quality.coverage_declared",
                    "quality.feature_eligible",
                    "quality.price_complete",
                    "quality.volume_known",
                    "quality.scope_known",
                ):
                    continue
                frame[name] = frame[name].map(
                    lambda value: (
                        {"true": True, "false": False}.get(value.lower(), value)
                        if isinstance(value, str)
                        else value
                    )
                )
    else:
        raise ValueError("Expected a daily .csv or .parquet file")
    return normalize_daily_bars(frame, schedule=schedule, **normalization_options)


def _source_frame(frame, timestamp_column):
    if isinstance(frame, pd.DataFrame) and timestamp_column not in frame:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.copy()
            frame[timestamp_column] = frame.index
    return _frame(frame, OHLCV, "Daily bars")


def _metadata(
    out,
    *,
    instrument_id,
    entity_id,
    source_id,
    currency,
    price_adjustment,
    volume_adjustment,
    session_scope,
    ingested_at,
    availability_basis,
):
    defaults = dict(
        instrument_id=instrument_id,
        entity_id=entity_id,
        source_id=source_id,
        currency=currency,
        price_adjustment=price_adjustment,
        volume_adjustment=volume_adjustment,
        session_scope=session_scope,
        ingested_at=ingested_at,
        availability_basis=availability_basis,
    )
    for name, value in defaults.items():
        if name not in out:
            out[name] = value
    for name, value in (("revision_id", None), ("schema_version", "1")):
        if name not in out:
            out[name] = value
    _identifiers(
        out,
        ("instrument_id", "entity_id", "source_id", "currency", "revision_id", "schema_version"),
        nullable=("entity_id", "currency", "revision_id"),
    )
    for name, choices in (
        ("price_adjustment", PRICE_ADJUSTMENTS),
        ("volume_adjustment", VOLUME_ADJUSTMENTS),
        ("session_scope", SESSION_SCOPES),
    ):
        if not out[name].isin(choices).all():
            raise ValueError(f"{name} must be one of {choices}")


def _numbers(out):
    for name in OHLCV:
        if out[name].map(lambda value: isinstance(value, (bool, np.bool_, complex))).any():
            raise ValueError("OHLCV requires real numbers, not booleans or complex values")
        out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
        if np.isinf(out[name]).any():
            raise ValueError("OHLCV values must be finite or explicitly missing")
        if (out[name] < 0 if name == "volume" else out[name] <= 0).any():
            raise ValueError("Prices must be positive and volume non-negative")
    known_high = out.high.notna()
    known_low = out.low.notna()
    if (
        out.loc[known_high, "high"] < out.loc[known_high, ["open", "low", "close"]].max(axis=1)
    ).any():
        raise ValueError("Invalid daily high")
    if (
        out.loc[known_low, "low"] > out.loc[known_low, ["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Invalid daily low")


def _boolean_column(out, name, *, default, nullable=False):
    if name not in out:
        out[name] = default
    for value in out[name]:
        if nullable and _missing(value):
            continue
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} requires booleans")
    out[name] = out[name].astype("boolean" if nullable else bool)


def normalize_daily_bars(
    frame,
    *,
    schedule,
    timestamp_column="timestamp",
    label="session_date",
    instrument_id=None,
    entity_id=None,
    source_id="unknown",
    currency=None,
    price_adjustment="unknown",
    volume_adjustment="unknown",
    session_scope="unknown",
    ingested_at=None,
    availability_basis="receipt",
    latency="0s",
):
    """Validate one bar per instrument/exchange session, without price filling.

    ``label='session_date'`` reads a session_date column if present, otherwise
    timestamp_column/date-labelled midnight index. Such a label is an economic
    date, not a midnight publication. ``open``/``close`` require aware timestamps
    exactly matching schedule bounds. Holidays and dates outside the supplied
    calendar are rejected. Output timestamp/session_open/session_close use actual
    UTC exchange times; session_date remains an ISO economic date.

    instrument_id is required in rows or as argument; entity_id and currency are
    nullable, never guessed from ticker. Adjustment and scope metadata are
    declarations, not inferred from prices. Missing OHLCV stays NaN with quality
    masks; infinities and invalid known prices/volume/order are rejected.

    Availability cannot precede session close or known publication. Receipt mode
    requires aware ingestion. Without explicit available_at, derive it from real
    close + latency and later ingestion/publication as applicable. Non-receipt
    modes describe an explicit historical assumption, not verified delivery.
    point_in_time defaults False. Native OHLCV alone does not prove full RTH
    coverage: quality.session_complete is unknown unless supplied separately.
    The API does not infer listings/delistings or extend coverage beyond rows.
    """
    calendar = normalize_session_schedule(schedule)
    out = _source_frame(frame, timestamp_column)
    if label not in ("session_date", "open", "close"):
        raise ValueError("label must be session_date, open or close")
    if label == "session_date" and "session_date" in out:
        dates = [_session_date(value) for value in out.session_date]
    elif timestamp_column in out:
        if label == "session_date":
            dates = [_session_date(value) for value in out[timestamp_column]]
        else:
            times = _timestamps(out[timestamp_column], timestamp_column)
            bounds = calendar.open_at if label == "open" else calendar.close_at
            lookup = dict(zip(bounds, calendar.index, strict=True))
            if not times.isin(bounds).all():
                raise ValueError("Daily timestamps must exactly match exchange session bounds")
            dates = [lookup[time] for time in times]
    elif out.empty:
        dates = []
    else:
        raise ValueError("Daily bars require session_date or a timestamp column/index")
    out["session_date"] = pd.Series(dates, dtype=object)
    if "observation_date" in out and not out.observation_date.equals(out.session_date):
        raise ValueError("Daily observation_date must equal its economic session_date")
    out["observation_date"] = out.session_date
    if not out.session_date.isin(calendar.index).all():
        raise ValueError("Daily bars include a non-session date or fall outside schedule coverage")
    selected = calendar.reindex(dates).reset_index(drop=True)
    for column, expected in (
        ("session_open", selected.open_at),
        ("session_close", selected.close_at),
    ):
        if column in out:
            supplied = _timestamps(out[column], column)
            if not supplied.equals(expected):
                raise ValueError(f"{column} disagrees with exchange schedule")
        out[column] = expected
    out["timestamp"] = out.session_open
    _metadata(
        out,
        instrument_id=instrument_id,
        entity_id=entity_id,
        source_id=source_id,
        currency=currency,
        price_adjustment=price_adjustment,
        volume_adjustment=volume_adjustment,
        session_scope=session_scope,
        ingested_at=ingested_at,
        availability_basis=availability_basis,
    )
    if out.duplicated(["instrument_id", "session_date"]).any():
        raise ValueError("Duplicate instrument/session; resolve vintages before daily calculation")
    _numbers(out)
    delay = pd.Timedelta(latency)
    if pd.isna(delay) or delay < pd.Timedelta(0):
        raise ValueError("latency requires a non-negative fixed duration")
    out["ingested_at"] = _timestamps(out.ingested_at, "ingested_at", nullable=True)
    if "published_at" not in out:
        out["published_at"] = pd.NaT
    out["published_at"] = _timestamps(out.published_at, "published_at", nullable=True)
    if "available_at" not in out:
        available = out.session_close + delay
        receipt = out.availability_basis.eq("receipt")
        for index in out.index:
            if receipt.loc[index] and pd.notna(out.ingested_at.loc[index]):
                available.loc[index] = max(available.loc[index], out.ingested_at.loc[index])
            if pd.notna(out.published_at.loc[index]):
                available.loc[index] = max(available.loc[index], out.published_at.loc[index])
        out["available_at"] = available
    _provenance(out)
    if (out.available_at < out.session_close).any():
        raise ValueError("Daily bars cannot be available before the real session close")
    _boolean_column(out, "quality.session_complete", default=pd.NA, nullable=True)
    _boolean_column(out, "quality.coverage_verified", default=False)
    known = out.loc[:, OHLCV].notna().all(axis=1)
    eligible = known & out["quality.session_complete"].fillna(True)
    if "quality.feature_eligible" in out:
        _boolean_column(out, "quality.feature_eligible", default=False)
        eligible &= out["quality.feature_eligible"]
    out["quality.feature_eligible"] = eligible.astype(bool)
    out["quality.price_complete"] = out.loc[:, OHLCV[:-1]].notna().all(axis=1)
    out["quality.volume_known"] = out.volume.notna()
    out["quality.scope_known"] = out.session_scope.ne("unknown")
    out = out.sort_values(["instrument_id", "session_date"], kind="stable", ignore_index=True)
    out.attrs.update(
        timestamp_semantics="session_open",
        available_at="explicit UTC availability, no earlier than session_close",
        calendar=calendar.attrs.get("calendar", "injected"),
        price_filling="none",
    )
    return out


def _session_grid(session, step):
    intervals = [(session.open_at, session.close_at)]
    if pd.notna(session.break_start_at):
        intervals = [
            (session.open_at, session.break_start_at),
            (session.break_end_at, session.close_at),
        ]
    values = []
    for first, last in intervals:
        if (last - first).value % step.value:
            raise ValueError("source_frequency must divide each active session interval")
        values.extend(pd.date_range(first, last, freq=step, inclusive="left"))
    return pd.DatetimeIndex(values, tz="UTC")


def _coverage(coverage, calendar):
    if coverage is None:
        return None
    out = _frame(coverage, ("instrument_id", "first_session", "last_session"), "Coverage")
    _identifiers(out, ("instrument_id",))
    if out.instrument_id.duplicated().any():
        raise ValueError("Coverage requires one interval per instrument")
    for column in ("first_session", "last_session"):
        out[column] = [_session_date(value) for value in out[column]]
        if not out[column].isin(calendar.index).all():
            raise ValueError("Coverage bounds must be sessions present in the schedule")
    if (out.last_session < out.first_session).any():
        raise ValueError("last_session cannot precede first_session")
    return out.set_index("instrument_id")


def aggregate_session_bars(
    frame,
    *,
    schedule,
    source_frequency="1min",
    timestamp_column="timestamp",
    instrument_id=None,
    entity_id=None,
    source_id="unknown",
    currency=None,
    price_adjustment="unknown",
    volume_adjustment="unknown",
    ingested_at=None,
    availability_basis="receipt",
    latency="0s",
    partial="keep",
    extended="exclude",
    coverage=None,
):
    """Aggregate actual RTH intervals, respecting breaks and missing source bars.

    Source indexes label bar opens; source_frequency must divide active calendar
    intervals. Outside-session rows (extended hours, holidays and breaks) are
    excluded with counts in attrs or rejected via extended='raise'. Off-grid RTH
    timestamps and duplicate instrument/timestamp rows are rejected.

    partial='keep' emits incomplete OHLCV with quality masks and missing sessions
    as all-NaN OHLCV. 'drop' excludes incomplete sessions; 'raise' rejects them.
    No prices or volume are filled. Explicit coverage is a DataFrame containing
    instrument_id/first_session/last_session; it bounds expected sessions and
    supports declared listing/delisting coverage. Without it, only the first to
    last observed RTH session is considered, and coverage is not declared. Neither
    mode invents later sessions or treats input end as evidence of delisting.

    Each output is available no earlier than actual close, relevant input receipt
    times and declared latency. event_time is an assumption with PIT False, not
    verified historical delivery. Aggregation completeness means source-grid
    completeness, not an independent guarantee that the provider tape is complete.
    """
    calendar = normalize_session_schedule(schedule)
    data = _source_frame(frame, timestamp_column)
    if timestamp_column not in data:
        raise ValueError("Intraday bars require a timestamp column or DatetimeIndex")
    data["timestamp"] = _timestamps(data[timestamp_column], timestamp_column)
    _metadata(
        data,
        instrument_id=instrument_id,
        entity_id=entity_id,
        source_id=source_id,
        currency=currency,
        price_adjustment=price_adjustment,
        volume_adjustment=volume_adjustment,
        session_scope="rth",
        ingested_at=ingested_at,
        availability_basis=availability_basis,
    )
    step = pd.Timedelta(source_frequency)
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("source_frequency requires a positive fixed duration")
    delay = pd.Timedelta(latency)
    if pd.isna(delay) or delay < pd.Timedelta(0):
        raise ValueError("latency requires a non-negative fixed duration")
    if partial not in ("keep", "drop", "raise") or extended not in ("exclude", "raise"):
        raise ValueError("partial must be keep/drop/raise and extended must be exclude/raise")
    declared = _coverage(coverage, calendar)
    if data.duplicated(["instrument_id", "timestamp"]).any():
        raise ValueError("Duplicate instrument/timestamp")
    for _, group in data.groupby("instrument_id", sort=False):
        normalize_bars(group, frequency="1ns")
    data["ingested_at"] = _timestamps(data.ingested_at, "ingested_at", nullable=True)
    if "published_at" not in data:
        data["published_at"] = pd.NaT
    data["published_at"] = _timestamps(data.published_at, "published_at", nullable=True)
    _boolean_column(data, "point_in_time", default=False)
    if "available_at" in data:
        data["available_at"] = _timestamps(data.available_at, "available_at")
        if (data.available_at < data.timestamp + step).any():
            raise ValueError("Intraday input cannot be available before its bar close")
    else:
        data["available_at"] = data.timestamp + step + delay
        for index in data.index:
            if data.availability_basis.loc[index] == "receipt" and pd.notna(
                data.ingested_at.loc[index]
            ):
                data.loc[index, "available_at"] = max(
                    data.available_at.loc[index], data.ingested_at.loc[index]
                )
            if pd.notna(data.published_at.loc[index]):
                data.loc[index, "available_at"] = max(
                    data.available_at.loc[index], data.published_at.loc[index]
                )
    _provenance(data)
    data = data.sort_values(["instrument_id", "timestamp"], kind="stable", ignore_index=True)
    opens, closes = calendar.open_at.array.asi8, calendar.close_at.array.asi8
    positions = np.searchsorted(opens, data.timestamp.array.asi8, side="right") - 1
    inside = positions >= 0
    if len(calendar):
        inside &= data.timestamp.array.asi8 < closes[np.maximum(positions, 0)]
        for index in np.flatnonzero(inside):
            session = calendar.iloc[positions[index]]
            if pd.notna(session.break_start_at):
                inside[index] &= not (
                    session.break_start_at <= data.timestamp.iloc[index] < session.break_end_at
                )
    else:
        inside[:] = False
    excluded = data.loc[~inside].groupby("instrument_id").size().to_dict()
    if extended == "raise" and not inside.all():
        raise ValueError("Input includes outside-session/extended-hours bars")
    data = data.loc[inside].copy()
    data["session_date"] = calendar.index.to_numpy()[positions[inside]]
    for session_date, group in data.groupby("session_date", sort=False):
        grid = _session_grid(calendar.loc[session_date], step)
        if not group.timestamp.isin(grid).all():
            raise ValueError("Intraday RTH bars must lie on the exchange session grid")
    instruments = set(data.instrument_id) | (set(declared.index) if declared is not None else set())
    records = []
    for instrument in sorted(instruments):
        supplied = data.loc[data.instrument_id == instrument]
        if declared is not None:
            if instrument not in declared.index:
                raise ValueError("Observed instrument is absent from declared coverage")
            first, last = declared.loc[instrument, ["first_session", "last_session"]]
            if ((supplied.session_date < first) | (supplied.session_date > last)).any():
                raise ValueError("RTH data falls outside declared instrument coverage")
        else:
            first, last = supplied.session_date.min(), supplied.session_date.max()
        instrument_schedule = calendar.loc[(calendar.index >= first) & (calendar.index <= last)]
        groups = {key: group for key, group in supplied.groupby("session_date", sort=False)}
        metadata = supplied.iloc[0].to_dict() if len(supplied) else {}
        for name in (
            "entity_id",
            "source_id",
            "currency",
            "price_adjustment",
            "volume_adjustment",
            "availability_basis",
        ):
            if supplied[name].nunique(dropna=False) > 1:
                raise ValueError(f"{name} must remain consistent within an aggregated instrument")
        for session_date, session in instrument_schedule.iterrows():
            bars = groups.get(session_date)
            observed = 0 if bars is None else len(bars)
            expected = len(_session_grid(session, step))
            complete = observed == expected
            if not complete and partial == "raise":
                raise ValueError("Incomplete exchange session coverage")
            if not complete and partial == "drop":
                continue
            record = dict(
                instrument_id=instrument,
                session_date=session_date,
                open=np.nan if bars is None else bars.open.iloc[0],
                high=np.nan if bars is None else bars.high.max(),
                low=np.nan if bars is None else bars.low.min(),
                close=np.nan if bars is None else bars.close.iloc[-1],
                volume=np.nan if bars is None else bars.volume.sum(),
                available_at=session.close_at + delay,
                point_in_time=False,
                **{
                    "quality.session_complete": complete,
                    "quality.coverage_verified": True,
                    "quality.coverage_declared": declared is not None,
                    "quality.observed_bars": observed,
                    "quality.expected_bars": expected,
                    "quality.missing_bars": expected - observed,
                },
            )
            for name in (
                "entity_id",
                "source_id",
                "currency",
                "price_adjustment",
                "volume_adjustment",
            ):
                if name in metadata:
                    record[name] = metadata[name]
            if bars is not None:
                record["available_at"] = max(session.close_at + delay, bars.available_at.max())
                record["availability_basis"] = bars.availability_basis.iloc[0]
                record["ingested_at"] = bars.ingested_at.max()
                record["published_at"] = bars.published_at.max()
                record["point_in_time"] = bool(complete and bars.point_in_time.all())
            elif metadata:
                record["availability_basis"] = metadata["availability_basis"]
                record["ingested_at"] = supplied.ingested_at.max()
                if metadata["availability_basis"] == "receipt" and pd.notna(record["ingested_at"]):
                    record["available_at"] = max(record["available_at"], record["ingested_at"])
            records.append(record)
    out = pd.DataFrame(records)
    if out.empty:
        out = pd.DataFrame(columns=["instrument_id", "session_date", *OHLCV])
    out = normalize_daily_bars(
        out,
        schedule=calendar,
        entity_id=entity_id,
        source_id=source_id,
        currency=currency,
        price_adjustment=price_adjustment,
        volume_adjustment=volume_adjustment,
        session_scope="rth",
        ingested_at=ingested_at,
        availability_basis=availability_basis,
        latency=latency,
    )
    out.attrs.update(
        excluded_extended_bars=excluded,
        source_frequency=str(step),
        coverage_declared=declared is not None,
    )
    return out
