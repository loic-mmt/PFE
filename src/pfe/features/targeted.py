"""Selected fine-to-coarse features, with vector reductions for common outputs."""

from functools import lru_cache

import numpy as np
import pandas as pd

from pfe.features._plan import Planner
from pfe.features.aggregation import fixed_duration
from pfe.features.backend import native, using_cpp
from pfe.features.intrabar import intrabar_features
from pfe.features.selected import _names, _validate, compute_selected_features

_BARS = {"open", "high", "low", "close", "volume"}
_FAST = _BARS | {
    "quality.observed_bars",
    "quality.expected_bars",
    "quality.missing_bars",
    "quality.coverage",
    "quality.complete",
    "quality.closed",
    "quality.observed_span_seconds",
    "path.realized_variance",
    "path.realized_volatility",
    "path.realized_variance_observed",
    "path.absolute_return_sum",
    "path.up_semivariance",
    "path.down_semivariance",
    "path.up_return_sum",
    "path.down_return_sum",
    "path.directional_efficiency",
}


@lru_cache(maxsize=32)
def _path_plan(duration, source, segments, bins, threshold, scales):
    return Planner(
        intrabar_features,
        dict(
            duration=duration,
            source_duration=source,
            segments=segments,
            bins=bins,
            threshold=threshold,
            internal_scales=scales,
        ),
        inputs=("bars", "start"),
    )


def aggregate_selected_features(
    bars,
    names,
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
    validate=True,
):
    """Compute exact target columns plus available_at; no full intrabar/profile pass.

    Popular OHLCV, realized variance and quality outputs use vectorized reductions.
    Other descriptors execute only their scalar dependency plan per target bin.
    Index/availability/partial/gap semantics match enrich_bars.
    """
    names = _names(names)
    source, duration = (
        fixed_duration(source_frequency, "source_frequency"),
        fixed_duration(frequency, "frequency"),
    )
    if duration < source or duration.value % source.value:
        raise ValueError("frequency must be an integer multiple of source_frequency")
    if partial not in {"drop", "keep", "raise"}:
        raise ValueError("partial must be 'drop', 'keep' or 'raise'")
    for value in (segments, bins):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("segments and bins must be positive integers")
    scales = tuple(internal_scales)
    if len(set(scales)) != len(scales) or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in scales
    ):
        raise ValueError("internal_scales must contain unique positive integers")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    slow = [name for name in names if name not in _FAST and name != "quality.crosses_local_date"]
    plan = _path_plan(duration, source, segments, bins, threshold, scales) if slow else None
    if plan is not None:
        for name in slow:
            if name not in plan.output:
                raise ValueError(f"Unknown feature: {name}")
    data, _ = _validate(bars, source, validate)
    latest = data.index[-1] + source
    cutoff = latest if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is None or cutoff > latest:
        raise ValueError("as_of must be aware and cannot exceed the last observed source close")
    cutoff = cutoff.tz_convert("UTC")
    count = int(data.index.searchsorted(cutoff - source, side="right"))
    data = data.iloc[:count]

    def empty():
        frame = pd.DataFrame(
            {
                **{name: pd.Series(dtype=float) for name in names},
                "available_at": pd.Series(dtype="datetime64[ns, UTC]"),
            },
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
        return frame

    if data.empty:
        return empty()
    if isinstance(origin, str) and origin in {"start_day", "epoch", "start"}:
        anchor = (
            data.index[0].normalize()
            if origin == "start_day"
            else (pd.Timestamp("1970-01-01", tz="UTC") if origin == "epoch" else data.index[0])
        )
    else:
        anchor = pd.Timestamp(origin)
        if anchor.tzinfo is None:
            raise ValueError("origin must be timezone-aware")
    anchor = anchor.value + (0 if offset is None else pd.Timedelta(offset).value)
    raw = data.index.asi8
    ids = (raw - anchor) // duration.value
    first = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    ends = np.r_[first[1:], len(data)]
    starts = anchor + ids[first] * duration.value
    if np.any((raw[first] - starts) % source.value):
        raise ValueError("origin/offset must align target bins with the source grid")
    del ids
    scheduled = starts + duration.value
    closed = scheduled <= cutoff.value
    edge = (starts < raw[0]) | ~closed
    if partial == "raise" and edge.any():
        raise ValueError("Truncated edge bin; use partial='drop' or 'keep'")
    kept = np.flatnonzero(~edge if partial == "drop" else np.ones(len(first), dtype=bool))
    observed = ends - first
    expected = int(duration / source)
    complete = (observed == expected) & (raw[first] == starts)
    out = {}
    returns = None
    return_names = [name for name in names if name.startswith("path.") and name in _FAST]
    if return_names:
        close, opening = data.close.to_numpy(), data.open.to_numpy()
        returns = np.empty(len(data))
        np.log(close, out=returns)
        # Transform in bounded slices so the shifted subtraction does not allocate n floats.
        for left in range(len(data) - 1, 0, -100_000):
            begin = max(1, left - 100_000 + 1)
            returns[begin : left + 1] -= returns[begin - 1 : left].copy()
        returns[first] = np.log(close[first] / opening[first])
        gaps = np.flatnonzero(np.diff(raw) != source.value) + 1
        gaps = gaps[
            (raw[gaps] - anchor) // duration.value == (raw[gaps - 1] - anchor) // duration.value
        ]
        returns[gaps] = 0.0
    shared = {}

    def reduce(key, values, operation=np.add):
        if key not in shared:
            shared[key] = operation.reduceat(values() if callable(values) else values, first)
        return shared[key]

    for name in names:
        if name in {"open", "close"}:
            value = data[name].to_numpy()[first if name == "open" else ends - 1]
        elif name in {"high", "low", "volume"}:
            operation = np.maximum if name == "high" else np.minimum if name == "low" else np.add
            value = operation.reduceat(data[name].to_numpy(), first)
        elif name == "quality.observed_bars":
            value = observed.astype(float)
        elif name == "quality.expected_bars":
            value = np.full(len(first), expected, dtype=float)
        elif name == "quality.missing_bars":
            value = np.maximum(0, expected - observed).astype(float)
        elif name == "quality.coverage":
            value = observed / expected
        elif name == "quality.complete":
            value = complete.astype(float)
        elif name == "quality.closed":
            value = closed.astype(float)
        elif name == "quality.observed_span_seconds":
            value = (raw[ends - 1] - raw[first] + source.value) / 1e9
        elif name == "quality.crosses_local_date":
            local = data.index.tz_convert(session_timezone)
            value = np.asarray(local[first].date != local[ends - 1].date, dtype=float)
        elif name in return_names:
            if "variance" in name or name == "path.realized_volatility":
                signal = (
                    returns
                    if "up_" not in name and "down_" not in name
                    else (np.maximum(returns, 0) if "up_" in name else np.minimum(returns, 0))
                )
                value = reduce(
                    "up" if "up_" in name else "down" if "down_" in name else "rv",
                    lambda signal=signal: signal * signal,
                )
                if name in {"path.realized_variance", "path.realized_volatility"}:
                    value = np.where(complete, value, np.nan)
                if name == "path.realized_volatility":
                    value = np.sqrt(value)
            elif name == "path.up_return_sum":
                value = reduce("up_sum", lambda: np.maximum(returns, 0))
            elif name == "path.down_return_sum":
                value = reduce("down_sum", lambda: np.minimum(returns, 0))
            elif name == "path.absolute_return_sum":
                value = reduce("length", lambda: np.abs(returns))
            else:
                length = reduce("length", lambda: np.abs(returns))
                value = np.divide(
                    np.abs(reduce("sum", returns)),
                    length,
                    out=np.full(len(first), np.nan),
                    where=length != 0,
                )
        else:
            continue
        out[name] = np.asarray(value)[kept]
    if slow:
        # Keep NumPy's platform-specific argsort tie ordering for this one metric.
        compiled = (
            [name for name in slow if name != "path.top3_moves_volume_share"] if using_cpp() else []
        )
        values = {}
        if compiled:
            matrix = native.intrabar_batch(
                *(
                    data[field].to_numpy(dtype=float, copy=False)
                    for field in ("open", "high", "low", "close", "volume")
                ),
                raw,
                first[kept],
                ends[kept],
                starts[kept],
                duration.value,
                source.value,
                bins,
                segments,
                threshold,
                compiled,
            )
            values.update({name: matrix[i] for i, name in enumerate(compiled)})
        reference = [name for name in slow if name not in compiled]
        values.update({name: np.empty(len(kept)) for name in reference})
        if reference:
            for row, group_id in enumerate(kept):
                block = data.iloc[first[group_id] : ends[group_id]]
                calculated = plan.evaluate(
                    reference, bars=block, start=pd.Timestamp(starts[group_id], tz="UTC")
                )
                for name, value in calculated.items():
                    values[name][row] = value
        out.update(values)
    result = pd.DataFrame(
        {
            **{name: out[name] for name in names},
            "available_at": pd.DatetimeIndex(scheduled[kept], tz="UTC"),
        },
        index=pd.DatetimeIndex(starts[kept], tz="UTC", name="timestamp"),
        copy=False,
    )
    result.attrs.update(
        frequency=str(duration),
        source_frequency=str(source),
        segments=segments,
        bins=bins,
        timestamp_semantics="bar_open",
        availability_column="available_at",
    )
    return result


def build_selected_features(
    bars,
    names,
    frequency,
    *,
    source_frequency="1min",
    gap_policy="reset",
    validate=True,
    standard_options=None,
    **aggregation_options,
):
    """Selected source-path and target OHLCV features, plus explicit availability."""
    names = _names(names)
    path = [
        name
        for name in names
        if name.split(".", 1)[0] in {"path", "profile", "quality"} or name in _BARS
    ]
    standard = [name for name in names if name not in path]
    if aggregation_options.get("partial", "drop") != "drop":
        raise ValueError("Target OHLCV features require partial='drop'")
    selected = list(dict.fromkeys([*path, *(sorted(_BARS) if standard else [])]))
    target = aggregate_selected_features(
        bars,
        selected,
        frequency,
        source_frequency=source_frequency,
        validate=validate,
        **aggregation_options,
    )
    if standard:
        options = {
            "session_timezone": aggregation_options.get("session_timezone", "America/New_York"),
            **(standard_options or {}),
        }
        frame = compute_selected_features(
            target if not target.empty else bars.iloc[:1],
            standard,
            source_frequency=frequency,
            gap_policy=gap_policy,
            validate=False,
            **options,
        )
        target = target.join(frame if not target.empty else frame.iloc[:0])
    return target[[*names, "available_at"]]
