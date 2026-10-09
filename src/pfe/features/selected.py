"""Exact-column APIs: evaluate dependencies, not entire feature families."""

import re
from functools import lru_cache

import numpy as np
import pandas as pd
import talib
from talib import abstract

from pfe.features._plan import Planner
from pfe.features._utils import validate_windows
from pfe.features.aggregation import fixed_duration
from pfe.features.backend import native, using_cpp
from pfe.features.complexity import describe_window
from pfe.features.rolling import COMPLEXITY_CPP
from pfe.features.sessions import session_features
from pfe.features.standard import standard_features
from pfe.features.structures import structure_features
from pfe.features.technical import TA_GROUPS, extensions


def _names(names):
    values = (names,) if isinstance(names, str) else tuple(names)
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise ValueError("Select at least one feature name")
    if len(set(values)) != len(values):
        raise ValueError("Feature names must be unique")
    return values


def _windows(names, fallback=(5, 20, 60)):
    windows = sorted({int(w) for name in names for w in re.findall(r"_w(\d+)", name)})
    return validate_windows(tuple(windows) if windows else fallback)


def _parameters(function, window):
    params = dict(function.parameters)
    if window is None:
        return params
    for key in params:
        if "period" in key:
            params[key] = window
    for key in ("fastperiod", "fastk_period"):
        if key in params:
            params[key] = max(2, window // 2)
    if "slowperiod" in params:
        params["slowperiod"] = max(3, window)
    for key in ("signalperiod", "slowk_period", "slowd_period", "fastd_period"):
        if key in params:
            params[key] = max(2, window // 3)
    for key, value in (
        ("timeperiod1", max(2, window // 2)),
        ("timeperiod2", window),
        ("timeperiod3", 2 * window),
        ("minperiod", 2),
    ):
        if key in params:
            params[key] = value
    return params


@lru_cache(maxsize=32)
def _native_catalog(windows):
    catalog = {}
    groups = talib.get_function_groups()
    for group in (*TA_GROUPS, "Pattern Recognition"):
        patterns = group == "Pattern Recognition"
        for name in groups[group]:
            function = abstract.Function(name)
            adjustable = any("period" in key for key in function.parameters)
            for window in windows if adjustable and not patterns else (None,):
                for label in function.output_names:
                    suffix = "" if window is None else f"_w{window}"
                    output = "" if len(function.output_names) == 1 else f"_{label}"
                    key = (
                        f"{'patterns' if patterns else 'technical'}.{name.lower()}{output}{suffix}"
                    )
                    catalog[key] = (name, label, window, None)
                    if patterns:
                        catalog[f"{key}_age"] = (name, label, None, "age")
                        for period in windows:
                            catalog[f"{key}_frequency_w{period}"] = (name, label, None, period)
    return catalog


@lru_cache(maxsize=64)
def _plan(family, windows, threshold=0.01, session_options=()):
    if family in {"calendar", "session"}:
        return Planner(session_features, dict(session_options))
    if family == "technical":
        from pfe.features._kernels import adaptive_mean, wma

        return Planner(
            extensions, dict(windows=windows), overrides=dict(adaptive_mean=adaptive_mean, wma=wma)
        )
    if family == "structure":
        return Planner(structure_features, dict(windows=windows, threshold=threshold))
    return Planner(standard_features, dict(windows=windows, groups={family}))


@lru_cache(maxsize=32)
def _complexity_plan():
    return Planner(describe_window, {}, inputs=("x",))


def _complexity(data, names):
    plan = _complexity_plan()
    out = {}
    tasks = {}
    for name in names:
        match = re.fullmatch(r"complexity\.(return|price|volume)_(.+)_w(\d+)", name)
        if match is None or match[2] not in plan.output:
            raise ValueError(f"Unknown feature: {name}")
        signal, statistic, window = match[1], match[2], int(match[3])
        tasks.setdefault((signal, window), []).append((name, statistic))
    for (signal, window), selected in tasks.items():
        source = np.log1p(data.volume) if signal == "volume" else np.log(data.close)
        if signal == "return":
            source = source.diff()
        # Common distribution descriptors use native rolling kernels directly.
        direct = {
            "q05": lambda source=source, window=window: source.rolling(window).quantile(0.05),
            "q95": lambda source=source, window=window: source.rolling(window).quantile(0.95),
        }
        scalar = [(name, key) for name, key in selected if key not in direct]
        compiled = (
            [(name, key) for name, key in scalar if key in COMPLEXITY_CPP] if using_cpp() else []
        )
        if compiled:
            matrix = native.rolling(
                source.to_numpy(dtype=float, copy=False),
                window,
                [key for _, key in compiled],
                np.empty(0),
            )
            out.update({name: matrix[i] for i, (name, _) in enumerate(compiled)})
            scalar = [item for item in scalar if item not in compiled]
        for name, key in selected:
            if key in direct:
                out[name] = direct[key]()
        if scalar:
            # One window traversal and one shared dependency context per window.
            values = source.to_numpy()
            outputs = {name: np.full(len(data), np.nan) for name, _ in scalar}
            keys = [key for _, key in scalar]
            for end in range(window - 1, len(data)):
                x = values[end - window + 1 : end + 1]
                if np.isfinite(x).all():
                    result = plan.evaluate(keys, x=x)
                    for name, key in scalar:
                        outputs[name][end] = result[key]
            out.update(outputs)
    return out


def _validate(bars, step, validate=True):
    """Bound temporary validation memory; never sort/copy all five input columns."""
    if not isinstance(bars, pd.DataFrame) or bars.empty or not bars.columns.is_unique:
        raise ValueError("Expected a non-empty DataFrame with unique columns")
    if set(("open", "high", "low", "close", "volume")) - set(bars.columns):
        raise ValueError("Expected OHLCV columns; use normalize_bars first")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Expected a timezone-aware DatetimeIndex; use normalize_bars first")
    index = bars.index.tz_convert("UTC")
    if index.unit != "ns":
        index = index.as_unit("ns")
    data = bars if bars.index.equals(index) else bars.set_axis(index, copy=False)
    gaps = [0]
    raw = index.asi8
    first = raw[0]
    for left in range(0, len(bars), 100_000):
        right = min(left + 100_000, len(bars))
        times = raw[max(0, left - 1) : right]
        if np.any(times == pd.NaT.value):
            raise ValueError("Timestamps must not be missing")
        differences = np.diff(times)
        if np.any(differences <= 0):
            raise ValueError("Timestamps must be unique and sorted")
        if np.any((raw[left:right] - first) % step.value):
            raise ValueError("Source timestamps must lie on a source_frequency grid")
        offset = max(0, left - 1)
        gaps.extend((np.flatnonzero(differences != step.value) + offset + 1).tolist())
        if validate:
            try:
                o, h, low, c, v = (
                    data[name].iloc[left:right].to_numpy()
                    for name in ("open", "high", "low", "close", "volume")
                )
                if any(not np.isfinite(value).all() for value in (o, h, low, c, v)):
                    raise ValueError("OHLCV values must be finite")
                if any(np.any(value <= 0) for value in (o, h, low, c)) or np.any(v < 0):
                    raise ValueError("Prices must be positive; volume non-negative")
                if np.any(h < np.maximum.reduce((o, low, c))) or np.any(
                    low > np.minimum.reduce((o, h, c))
                ):
                    raise ValueError("Invalid OHLC price bounds")
            except TypeError as exc:
                raise ValueError("Expected numeric OHLCV; use normalize_bars first") from exc
    gaps.append(len(bars))
    return data, gaps


def _native(data, specs):
    from pfe.features._utils import event_age

    out, cache = {}, {}
    for key, (name, label, window, transform) in specs.items():
        cache_key = (name, window)
        if cache_key not in cache:
            function = abstract.Function(name)
            function.set_parameters(_parameters(function, window))
            required = set()
            for inputs in function.input_names.values():
                required.update([inputs] if isinstance(inputs, str) else inputs)
            values = {
                field: data[field].to_numpy(dtype=float, copy=False)
                for field in required
                if field in data
            }
            if name == "MAVP":
                values["periods"] = np.full(len(data), window or 14, dtype=float)
            result = function(values)
            arrays = result if isinstance(result, (tuple, list)) else [result]
            output = {}
            for field, values in zip(function.output_names, arrays, strict=True):
                array = np.asarray(values, dtype=float)
                array[: function.lookback] = np.nan
                output[field] = pd.Series(array, index=data.index)
            cache[cache_key] = output
        series = cache[cache_key][label]
        if transform is not None:
            event = (series != 0).where(series.notna(), False)
            series = (
                event_age(event).where(series.notna())
                if transform == "age"
                else (event.astype(float).where(series.notna()).rolling(transform).mean())
            )
        out[key] = series
    return out


def compute_selected_features(
    bars: pd.DataFrame,
    names,
    *,
    source_frequency: str | pd.Timedelta = "1min",
    gap_policy: str = "reset",
    validate: bool = True,
    frequency: str | pd.Timedelta | None = None,
    session_timezone: str = "America/New_York",
    session_start: str = "09:30",
    session_end: str = "16:00",
    opening_range: str = "15min",
    structure_threshold: float = 0.01,
    **aggregation_options,
) -> pd.DataFrame:
    """Compute exact named columns with shared dependencies and no group fallback.

    Names match the existing catalog, including windows (technical.rsi_w14).
    Input must already be sorted numeric OHLCV with an aware index. Validation
    uses bounded scratch memory; validate=False skips numeric checks only, for
    trusted normalized inputs. Gap resets and availability match compute_features.
    """
    if frequency is not None:
        from pfe.features.targeted import build_selected_features

        return build_selected_features(
            bars,
            names,
            frequency,
            source_frequency=source_frequency,
            gap_policy=gap_policy,
            validate=validate,
            session_timezone=session_timezone,
            standard_options=dict(
                session_start=session_start,
                session_end=session_end,
                opening_range=opening_range,
                structure_threshold=structure_threshold,
            ),
            **aggregation_options,
        )
    if aggregation_options:
        raise TypeError("Aggregation options require frequency")
    names = _names(names)
    windows = _windows(names)
    step = fixed_duration(source_frequency, "source_frequency")
    if gap_policy not in {"reset", "allow"}:
        raise ValueError("gap_policy must be 'reset' or 'allow'")
    if not np.isfinite(structure_threshold) or not 0 < structure_threshold < 1:
        raise ValueError("structure_threshold must lie strictly between 0 and 1")
    options = tuple(
        dict(
            source_duration=step,
            timezone=session_timezone,
            session_start=session_start,
            session_end=session_end,
            opening_range=opening_range,
        ).items()
    )
    native = _native_catalog(windows)
    tasks, native_names, complexity_names = {}, {}, []
    for name in names:
        if name in native:
            native_names[name] = native[name]
        elif name.startswith("complexity."):
            complexity_names.append(name)
        else:
            family = name.split(".", 1)[0]
            if family not in {
                "price",
                "returns",
                "candles",
                "statistics",
                "volatility",
                "trend",
                "volume",
                "levels",
                "sequences",
                "technical",
                "structure",
                "calendar",
                "session",
            }:
                raise ValueError(f"Unknown feature: {name}")
            configured = _windows((name,)) if len(re.findall(r"_w\d+", name)) > 1 else windows
            plan = _plan(
                "calendar" if family == "session" else family,
                configured,
                structure_threshold,
                options,
            )
            if name not in plan.output:
                raise ValueError(f"Unknown feature: {name}")
            tasks.setdefault(plan, []).append(name)
    if complexity_names:
        # Validate names before reading the actual dataset.
        plan = _complexity_plan()
        for name in complexity_names:
            match = re.fullmatch(r"complexity\.(return|price|volume)_(.+)_w(\d+)", name)
            if match is None or match[2] not in plan.output:
                raise ValueError(f"Unknown feature: {name}")
    data, boundaries = _validate(bars, step, validate)
    if gap_policy == "allow":
        boundaries = [0, len(data)]
    contiguous = len(boundaries) == 2
    output = {} if contiguous else {name: np.empty(len(data), dtype=float) for name in names}
    session_plans = {
        plan for plan in tasks if any(n.startswith(("calendar.", "session.")) for n in tasks[plan])
    }
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        block = data if left == 0 and right == len(data) else data.iloc[left:right]
        calculated = _native(block, native_names) if native_names else {}
        for plan, selected in tasks.items():
            if plan not in session_plans:
                calculated.update(plan.evaluate(selected, bars=block))
        if complexity_names:
            calculated.update(_complexity(block, complexity_names))
        for name, value in calculated.items():
            if contiguous:
                output[name] = np.asarray(value, dtype=float)
            else:
                output[name][left:right] = np.asarray(value)
    for plan in session_plans:
        for name, value in plan.evaluate(tasks[plan], bars=data).items():
            output[name] = np.asarray(value, dtype=float)
    for values in output.values():
        for left in range(0, len(values), 100_000):
            block = values[left : left + 100_000]
            block[np.isinf(block)] = np.nan
    output = {name: output[name] for name in names}
    result = pd.DataFrame(output, index=data.index, copy=False)
    result.attrs.update(
        source_frequency=str(step),
        timestamp_semantics="bar_open",
        available_at="index + source_frequency",
        gap_policy=gap_policy,
        selected_features=names,
    )
    return result


def compute_feature(
    bars: pd.DataFrame, name: str, *, window: int | None = None, **options
) -> pd.Series:
    """Compute one exact feature; optional window turns technical.rsi into rsi_w14."""
    if window is not None:
        validate_windows((window,))
        if re.search(r"_w\d+", name):
            raise ValueError("Specify the window in the name or window=, not both")
        name = f"{name}_w{window}"
    frame = compute_selected_features(bars, (name,), **options)
    result = frame[name]
    result.attrs.update(frame.attrs)
    if options.get("frequency") is not None:
        result.attrs.update(
            source_frequency=str(fixed_duration(options["frequency"], "frequency")),
            available_at="index + source_frequency",
        )
    return result
