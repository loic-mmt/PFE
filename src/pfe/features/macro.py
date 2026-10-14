"""Sparse macro transformations and causal alignment to market bar closes."""

import heapq
import re
from bisect import insort

import numpy as np
import pandas as pd

from pfe.data.macro import normalize_macro_calendar, normalize_macro_series
from pfe.features.aggregation import fixed_duration

MACRO_FEATURE_GROUPS = ("series", "calendar", "surprise", "spreads")
_SERIES = (
    "level",
    "change",
    "pct_change",
    "age_seconds",
    "observation_age_seconds",
    "revision",
    "is_revision",
    "point_in_time",
)
_CALENDAR = (
    "seconds_to_next_event",
    "seconds_since_event",
    "events_next_window",
    "events_previous_window",
    "event_now",
)
_EVENT = (
    "actual",
    "forecast",
    "previous",
    "surprise",
    "surprise_relative",
    "actual_change",
    "age_seconds",
    "point_in_time",
)
_CHUNK = 262144


def _selection(value, label):
    if isinstance(value, str):
        value = (value,)
    result = tuple(value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def _ns(values):
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def _align(times, values, decisions, *, reference=None, max_age=None, age=False):
    """Backward lookup, with bounded temporary arrays regardless of market row count."""
    out = np.full(len(decisions), np.nan)
    for left in range(0, len(out), _CHUNK):
        right = min(left + _CHUNK, len(out))
        now = decisions[left:right]
        positions = np.searchsorted(times, now, side="right") - 1
        valid = positions >= 0
        where = positions[valid]
        if age:
            result = (now[valid] - values[where]) / 1e9
        else:
            result = values[where].astype(float, copy=True)
        if max_age is not None:
            result[now[valid] - reference[where] > max_age] = np.nan
        chunk = out[left:right]
        chunk[valid] = result
    return out


def _series_history(frame, metrics):
    """Replay vintages on sparse observations, never on the dense market index."""
    periods, values, stamps, qualities = [], {}, {}, {}
    histories = {name: [] for name in metrics}
    times, latest_available = [], []
    for row in frame.itertuples(index=False):
        period = row.observation_at.value
        previous_value = values.get(period, np.nan)
        revised = period in values and not (
            previous_value == row.value or np.isnan(previous_value) and np.isnan(row.value)
        )
        if period not in values:
            insort(periods, period)
        values[period], stamps[period], qualities[period] = (
            row.value,
            row.available_at.value,
            float(row.point_in_time),
        )
        current = periods[-1]
        level = values[current]
        previous = values[periods[-2]] if len(periods) > 1 else np.nan
        times.append(row.available_at.value)
        latest_available.append(stamps[current])
        cache = {}
        for metric in metrics:
            if metric == "level":
                result = level
            elif metric == "change":
                result = level - previous
            elif metric == "pct_change":
                result = level / previous - 1 if previous != 0 else np.nan
            elif metric == "age_seconds":
                result = stamps[current]
            elif metric == "observation_age_seconds":
                result = current
            elif metric == "revision":
                result = row.value - previous_value if revised else 0.0
            elif metric == "is_revision":
                result = float(revised)
            elif metric == "point_in_time":
                result = qualities[current]
            else:
                match = re.fullmatch(r"(mean|std|zscore|slope)_w(\d+)", metric)
                kind, window = match[1], int(match[2])
                if len(periods) < window:
                    result = np.nan
                else:
                    if window not in cache:
                        cache[window] = np.array([values[p] for p in periods[-window:]])
                    sample = cache[window]
                    if kind == "mean":
                        result = sample.mean()
                    elif kind == "std":
                        result = sample.std(ddof=0)
                    elif kind == "zscore":
                        std = sample.std(ddof=0)
                        result = (level - sample.mean()) / std if std > 0 else np.nan
                    else:
                        x = np.arange(window, dtype=float) - (window - 1) / 2
                        result = np.dot(x, sample - sample.mean()) / np.dot(x, x)
            histories[metric].append(result)
    return (
        np.asarray(times, dtype=np.int64),
        {
            name: np.asarray(rows, dtype=np.int64 if name.endswith("age_seconds") else float)
            for name, rows in histories.items()
        },
        np.asarray(latest_available, dtype=np.int64),
    )


def _schedule(frame, decisions, metric, window):
    scheduled, known = _ns(frame.scheduled_at), _ns(frame.known_at)
    if metric in ("events_next_window", "events_previous_window", "event_now"):
        if metric == "event_now":
            times = np.sort(scheduled[known <= scheduled])
            return _counts(times, times, decisions, "left")
        if metric == "events_next_window":
            start, end = np.maximum(known, scheduled - window), scheduled
            valid, end_side = start < end, "right"
        else:
            start, end = np.maximum(known, scheduled), scheduled + window
            valid, end_side = start <= end, "left"
        starts, ends = np.sort(start[valid]), np.sort(end[valid])
        return _counts(starts, ends, decisions, end_side)
    if metric == "seconds_since_event":
        active = np.maximum(known, scheduled)
        order = np.argsort(active, kind="stable")
        latest = np.maximum.accumulate(scheduled[order])
        return _align(active[order], latest, decisions, age=True)
    # A future event is visible only after its schedule becomes known.
    times = np.unique(np.r_[known, scheduled])
    order = np.argsort(known, kind="stable")
    heap, cursor, next_times = [], 0, []
    for time in times:
        while cursor < len(order) and known[order[cursor]] <= time:
            heapq.heappush(heap, scheduled[order[cursor]])
            cursor += 1
        while heap and heap[0] <= time:
            heapq.heappop(heap)
        next_times.append(heap[0] if heap else np.nan)
    return -_align(times, np.asarray(next_times), decisions, age=True)


def _counts(starts, ends, decisions, end_side):
    out = np.empty(len(decisions), dtype=float)
    for left in range(0, len(out), _CHUNK):
        right = min(left + _CHUNK, len(out))
        now = decisions[left:right]
        out[left:right] = np.searchsorted(starts, now, side="right") - np.searchsorted(
            ends, now, side=end_side
        )
    return out


def _event_history(frame, metric):
    if metric in ("actual", "forecast", "previous"):
        present = frame[metric].notna()
        subset = frame.loc[present].sort_values(f"{metric}_available_at", kind="stable")
        return _ns(subset[f"{metric}_available_at"]), subset[metric].to_numpy(float)
    fields = ["actual"]
    if metric.startswith("surprise") or metric == "point_in_time":
        fields.append("forecast")
    elif metric == "actual_change":
        fields.append("previous")
    subset = frame.loc[frame[fields].notna().all(axis=1)].copy()
    time = np.maximum.reduce([_ns(subset[f"{field}_available_at"]) for field in fields])
    order = np.argsort(time, kind="stable")
    subset, time = subset.iloc[order], time[order]
    if metric == "age_seconds":
        return time, time
    if metric == "actual_change":
        return time, (subset.actual - subset.previous).to_numpy(float)
    if metric == "point_in_time":
        # A forecast acquired after the scheduled release is not a pre-release benchmark.
        return time, (
            subset.point_in_time & (subset.forecast_available_at <= subset.scheduled_at)
        ).to_numpy(float)
    surprise = (subset.actual - subset.forecast).to_numpy(float)
    if metric == "surprise":
        return time, surprise
    if metric == "surprise_relative":
        denominator = subset.forecast.abs().to_numpy(float)
        result = np.full(len(subset), np.nan)
        np.divide(surprise, denominator, out=result, where=denominator != 0)
        return time, result
    window = int(metric.rsplit("_w", 1)[1])
    # Scale each surprise with strictly earlier known surprises of the SAME event type.
    # Simultaneous batch receipts must not acquire an artificial chronological ordering.
    result = np.full(len(time), np.nan)
    for i, stamp in enumerate(time):
        end = np.searchsorted(time, stamp, side="left")
        if end >= window:
            previous = surprise[end - window : end]
            std = previous.std(ddof=0)
            if std > 0:
                result[i] = (surprise[i] - previous.mean()) / std
    return time, result


def compute_macro_features(
    bars,
    *,
    observations=None,
    events=None,
    calendar_history=None,
    names=None,
    groups=None,
    windows=(3, 12),
    pairs=(),
    source_frequency="1min",
    event_window="30min",
    max_age=None,
):
    """Compute selected macro features at bar CLOSE, preserving the bar-open index.

    Inputs are validated canonical frames; see ``normalize_macro_series/calendar``.
    Levels and rolling statistics use reference periods, not repeated market rows.
    ``max_age`` masks series/event values after their latest value receipt; calendar
    counts and explicit age features remain visible. Groups and names are exclusive.
    Pairs are ordered (left, right) series identifiers in compatible units.
    """
    index = bars.index if isinstance(bars, pd.DataFrame) else bars
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("bars require a sorted unique timezone-aware DatetimeIndex")
    step = fixed_duration(source_frequency, "source_frequency")
    width = fixed_duration(event_window, "event_window").value
    age_limit = fixed_duration(max_age, "max_age").value if max_age is not None else None
    windows = tuple(windows)
    if len(set(windows)) != len(windows) or any(
        isinstance(w, bool) or not isinstance(w, int) or w < 2 for w in windows
    ):
        raise ValueError("windows must be unique integers >= 2")
    series = normalize_macro_series(observations) if observations is not None else None
    if events is not None and calendar_history is not None:
        raise ValueError("Choose events or calendar_history")
    calendar_versions = None
    if calendar_history is not None:
        from pfe.data.macro_calendar import normalize_macro_calendar_history

        calendar_versions = normalize_macro_calendar_history(calendar_history)
    calendar = normalize_macro_calendar(events) if events is not None else None
    if series is None and calendar is None and calendar_versions is None:
        raise ValueError("Provide observations and/or events")
    symbols = tuple(series.series.unique()) if series is not None else ()
    calendar_input = calendar_versions if calendar_versions is not None else calendar
    event_types = tuple(calendar_input.event.unique()) if calendar_input is not None else ()
    catalog = {}
    for symbol in symbols:
        metrics = (
            *_SERIES,
            *(f"{m}_w{w}" for w in windows for m in ("mean", "std", "zscore", "slope")),
        )
        for metric in metrics:
            catalog[f"macro.{symbol}.{metric}"] = ("series", symbol, metric)
    if calendar_input is not None:
        for metric in _CALENDAR:
            catalog[f"macro.calendar.{metric}"] = ("calendar", None, metric)
        for event in event_types:
            for metric in (*_EVENT, *(f"surprise_zscore_w{w}" for w in windows)):
                catalog[f"macro.event[{event}].{metric}"] = ("surprise", event, metric)
    seen = set()
    for pair in pairs:
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(symbol not in symbols for symbol in pair)
            or pair[0] == pair[1]
            or tuple(pair) in seen
        ):
            raise ValueError("pairs require unique ordered pairs of distinct available series")
        pair = tuple(pair)
        seen.add(pair)
        for metric in ("spread", "ratio"):
            catalog[f"macro.pair[{pair[0]},{pair[1]}].{metric}"] = ("spreads", pair, metric)
    if names is not None and groups is not None:
        raise ValueError("Choose names or groups, not both")
    if names is not None:
        selected = _selection(names, "names")
        unknown = set(selected) - catalog.keys()
        if unknown:
            raise ValueError(f"Unknown macro features: {sorted(unknown)}")
    else:
        chosen = MACRO_FEATURE_GROUPS if groups is None else _selection(groups, "groups")
        if set(chosen) - set(MACRO_FEATURE_GROUPS):
            raise ValueError("Unknown macro feature group")
        selected = tuple(name for name, (group, _, _) in catalog.items() if group in chosen)
    decisions = _ns(index + step)
    needed = {}
    for name in selected:
        group, key, metric = catalog[name]
        if group == "series":
            needed.setdefault(key, set()).add(metric)
        elif group == "spreads":
            for symbol in key:
                needed.setdefault(symbol, set()).add("level")
    aligned = {}
    for symbol, metrics in needed.items():
        history, values, reference = _series_history(series.loc[series.series == symbol], metrics)
        for metric, value in values.items():
            is_age = metric.endswith("age_seconds")
            aligned[symbol, metric] = _align(
                history,
                value,
                decisions,
                reference=reference,
                max_age=None if is_age else age_limit,
                age=is_age,
            )
    output = {}
    if calendar_versions is not None:
        from pfe.features.macro_calendar import calendar_history_features

        output.update(
            calendar_history_features(
                calendar_versions,
                decisions,
                {
                    name: catalog[name]
                    for name in selected
                    if catalog[name][0] in ("calendar", "surprise")
                },
                window=width,
                max_age=age_limit,
            )
        )
    for name in selected:
        group, key, metric = catalog[name]
        if calendar_versions is not None and group in ("calendar", "surprise"):
            continue
        if group == "series":
            output[name] = aligned[key, metric]
        elif group == "calendar":
            output[name] = _schedule(calendar, decisions, metric, width)
        elif group == "surprise":
            times, value = _event_history(calendar.loc[calendar.event == key], metric)
            is_age = metric == "age_seconds"
            output[name] = _align(
                times,
                value,
                decisions,
                reference=times,
                max_age=None if is_age else age_limit,
                age=is_age,
            )
        else:
            left, right = (aligned[symbol, "level"] for symbol in key)
            if metric == "spread":
                output[name] = left - right
            else:
                result = np.full(len(index), np.nan)
                np.divide(left, right, out=result, where=right != 0)
                output[name] = result
    result = pd.DataFrame(output, index=index, copy=False).reindex(columns=selected)
    result.attrs.update(
        source_frequency=str(step),
        timestamp_semantics="bar_open",
        available_at="index + source_frequency",
    )
    return result


def compute_macro_feature(bars, name, **options):
    """Compute exactly one macro feature and its minimal shared prerequisites."""
    result = compute_macro_features(bars, names=(name,), **options)
    feature = result[name]
    feature.attrs.update(result.attrs)
    return feature
