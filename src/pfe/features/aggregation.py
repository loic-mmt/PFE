"""Configurable fixed-duration aggregation with explicit availability times."""

import numpy as np
import pandas as pd

from pfe.data import normalize_bars
from pfe.features.intrabar import intrabar_features


def fixed_duration(value, name: str) -> pd.Timedelta:
    try:
        step = pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a fixed duration such as '7min', '45min', '2h'") from exc
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError(f"{name} must be positive")
    return step


def enrich_bars(
    bars: pd.DataFrame,
    frequency: str | pd.Timedelta,
    *,
    source_frequency: str | pd.Timedelta = "1min",
    origin: str | pd.Timestamp = "start_day",
    offset: str | pd.Timedelta | None = None,
    as_of: str | pd.Timestamp | None = None,
    partial: str = "drop",
    segments: int = 5,
    bins: int = 10,
    threshold: float = 0.001,
    internal_scales: tuple[int, ...] = (1, 5, 10),
    session_timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Aggregate OHLCV and path/profile features at any fixed source multiple.

    Input index labels bar OPEN times. Output index also labels bar opens;
    ``available_at`` labels the CLOSE time when all features become usable.
    Bins are left-closed/right-open. ``partial='drop'`` removes truncated edge
    bins, but retains closed bins containing holes and reports their coverage.
    ``partial='keep'`` exposes provisional edge bins, flagged ``quality.closed=0``
    when their scheduled close has not arrived. ``as_of`` cannot exceed the last
    source bar's close: unobserved data is never declared complete.

    Equal-time segments use source-bar open timestamps. Price profiles are
    range/close proxies, never tick-level dwell time. All computations are local
    to a target bin, with no future target-bar information.
    """
    source = fixed_duration(source_frequency, "source_frequency")
    target = fixed_duration(frequency, "frequency")
    if target < source or target.value % source.value:
        raise ValueError("frequency must be an integer multiple of source_frequency")
    if partial not in {"drop", "keep", "raise"}:
        raise ValueError("partial must be 'drop', 'keep' or 'raise'")
    for name, value in (("segments", segments), ("bins", bins)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in internal_scales):
        raise ValueError("internal_scales must contain positive integers")
    if len(set(internal_scales)) != len(internal_scales):
        raise ValueError("internal_scales must be unique")
    data = normalize_bars(bars, frequency="1ns")
    if ((data.index - data.index[0]).asi8 % source.value != 0).any():
        raise ValueError("Source timestamps must lie on a source_frequency grid")
    latest = data.index[-1] + source
    sample = data.iloc[:1]
    cutoff = latest if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    if cutoff > latest:
        raise ValueError("as_of cannot exceed the last observed source bar close")
    data = data.loc[data.index + source <= cutoff]
    columns = ["open", "high", "low", "close", "volume", "available_at"]

    def empty_result():
        schema = intrabar_features(
            sample,
            start=sample.index[0],
            duration=target,
            source_duration=source,
            segments=segments,
            bins=bins,
            threshold=threshold,
            internal_scales=internal_scales,
        )
        names = [*columns, *schema, "quality.closed"]
        return pd.DataFrame(
            {
                name: pd.Series(dtype="datetime64[ns, UTC]" if name == "available_at" else float)
                for name in names
            },
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    if data.empty:
        return empty_result()
    records, starts = [], []
    for start, group in data.resample(
        target, origin=origin, offset=offset, closed="left", label="left"
    ):
        if group.empty:
            continue
        end = start + target
        if ((group.index - start).asi8 % source.value != 0).any():
            raise ValueError("origin/offset must align target bins with the source grid")
        closed = end <= cutoff
        edge_partial = start < data.index[0] or not closed
        if edge_partial and partial == "raise":
            raise ValueError("Truncated edge bin; use partial='drop' or 'keep'")
        if edge_partial and partial == "drop":
            continue
        row = {
            "open": float(group.open.iloc[0]),
            "high": float(group.high.max()),
            "low": float(group.low.min()),
            "close": float(group.close.iloc[-1]),
            "volume": float(group.volume.sum()),
            "available_at": end,
        }
        row.update(
            intrabar_features(
                group,
                start=start,
                duration=target,
                source_duration=source,
                segments=segments,
                bins=bins,
                threshold=threshold,
                internal_scales=internal_scales,
            )
        )
        row["quality.closed"] = float(closed)
        local = group.index.tz_convert(session_timezone)
        row["quality.crosses_local_date"] = float(local[0].date() != local[-1].date())
        records.append(row)
        starts.append(start)
    result = pd.DataFrame(records, index=pd.DatetimeIndex(starts, tz="UTC", name="timestamp"))
    if result.empty:
        result = empty_result()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    result.attrs.update(
        frequency=str(target),
        source_frequency=str(source),
        origin=str(origin),
        offset=str(offset),
        segments=segments,
        bins=bins,
        timestamp_semantics="bar_open",
        availability_column="available_at",
    )
    return result


_enrich_bars_python = enrich_bars


def enrich_bars(
    bars,
    frequency,
    *,
    source_frequency="1min",
    origin="start_day",
    offset=None,
    as_of=None,
    partial="drop",
    segments=5,
    bins=10,
    threshold=0.001,
    internal_scales=(1, 5, 10),
    session_timezone="America/New_York",
):
    """Existing group API; native batches preserve all columns and availability."""
    from pfe.features.backend import using_cpp

    options = dict(
        source_frequency=source_frequency,
        origin=origin,
        offset=offset,
        as_of=as_of,
        partial=partial,
        segments=segments,
        bins=bins,
        threshold=threshold,
        internal_scales=internal_scales,
        session_timezone=session_timezone,
    )
    if not using_cpp():
        return _enrich_bars_python(bars, frequency, **options)
    from pfe.features.targeted import _path_plan, aggregate_selected_features

    source, duration = (
        fixed_duration(source_frequency, "source_frequency"),
        fixed_duration(frequency, "frequency"),
    )
    for value in (segments, bins):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("segments and bins must be positive integers")
    # The schema comes from formula metadata, with no eager numeric group pass.
    plan = _path_plan(duration, source, segments, bins, threshold, tuple(internal_scales))
    names = ("open", "high", "low", "close", "volume", *plan.output, "quality.closed")
    data = normalize_bars(bars, frequency="1ns")
    result = aggregate_selected_features(data, names, frequency, validate=False, **options)
    ordered = intrabar_features(
        data.iloc[:1],
        start=data.index[0],
        duration=duration,
        source_duration=source,
        segments=segments,
        bins=bins,
        threshold=threshold,
        internal_scales=internal_scales,
    )
    result = result[
        ["open", "high", "low", "close", "volume", "available_at", *ordered, "quality.closed"]
    ]
    result.attrs.update(origin=str(origin), offset=str(offset))
    return result
