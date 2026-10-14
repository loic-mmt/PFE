"""Public feature APIs and an inspectable catalog of returned columns."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import talib
from talib import abstract

from pfe.data import normalize_bars
from pfe.features._utils import validate_windows
from pfe.features.aggregation import enrich_bars, fixed_duration
from pfe.features.complexity import complexity_features
from pfe.features.sessions import session_features
from pfe.features.standard import standard_features
from pfe.features.structures import structure_features
from pfe.features.technical import extensions, talib_features

FEATURE_GROUPS = (
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
    "patterns",
    "complexity",
    "structure",
    "calendar",
)


def _calculate(bars, windows, groups, structure_threshold):
    out = standard_features(bars, windows, groups)
    if "technical" in groups:
        out.update(talib_features(bars, windows))
        out.update(extensions(bars, windows))
    if "patterns" in groups:
        out.update(talib_features(bars, windows, patterns=True))
    if "complexity" in groups:
        out.update(complexity_features(bars, windows))
    if "structure" in groups:
        out.update(structure_features(bars, windows, structure_threshold))
    return pd.DataFrame(out, index=bars.index)


def compute_features(
    bars: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 20, 60),
    groups: Iterable[str] | None = None,
    names: Iterable[str] | None = None,
    source_frequency: str | pd.Timedelta = "1min",
    gap_policy: str = "reset",
    session_timezone: str = "America/New_York",
    session_start: str = "09:30",
    session_end: str = "16:00",
    opening_range: str = "15min",
    structure_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compute the OHLCV catalog using only completed current/past bars.

    Windows count observed bars. By default rolling/recursive state resets at
    every timestamp gap, preventing overnight gaps from becoming one-minute
    returns. ``gap_policy='allow'`` explicitly uses consecutive observations
    across gaps. Clock/session features are computed separately across dates.

    Output retains the input bar-open index. Values are usable at that open plus
    source_frequency. Zero-denominator features are NaN; no future labels or
    warm-up filling. ``groups`` selects families to control cost and width.
    """
    if names is not None:
        if groups is not None:
            raise ValueError("Choose groups or exact names, not both")
        from pfe.features.selected import compute_selected_features

        return compute_selected_features(
            bars,
            names,
            source_frequency=source_frequency,
            gap_policy=gap_policy,
            session_timezone=session_timezone,
            session_start=session_start,
            session_end=session_end,
            opening_range=opening_range,
            structure_threshold=structure_threshold,
        )
    windows = validate_windows(windows)
    if isinstance(groups, str):
        raise ValueError("groups must be an iterable of names, not a single string")
    selected = set(FEATURE_GROUPS if groups is None else groups)
    if selected - set(FEATURE_GROUPS):
        raise ValueError(f"Unknown feature groups: {selected - set(FEATURE_GROUPS)}")
    if gap_policy not in {"reset", "allow"}:
        raise ValueError("gap_policy must be 'reset' or 'allow'")
    if not np.isfinite(structure_threshold) or not 0 < structure_threshold < 1:
        raise ValueError("structure_threshold must lie strictly between 0 and 1")
    step = fixed_duration(source_frequency, "source_frequency")
    data = normalize_bars(bars, frequency="1ns")
    if ((data.index - data.index[0]).asi8 % step.value != 0).any():
        raise ValueError("Source timestamps must lie on a source_frequency grid")
    if gap_policy == "reset":
        ids = data.index.to_series().diff().ne(step).cumsum()
        chunks = [
            _calculate(group, windows, selected, structure_threshold)
            for _, group in data.groupby(ids)
        ]
        result = pd.concat(chunks).reindex(data.index)
    else:
        result = _calculate(data, windows, selected, structure_threshold)
    if "calendar" in selected:
        calendar = pd.DataFrame(
            session_features(
                data,
                step,
                timezone=session_timezone,
                session_start=session_start,
                session_end=session_end,
                opening_range=opening_range,
            ),
            index=data.index,
        )
        result = pd.concat([result, calendar], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    if not result.columns.is_unique:
        raise RuntimeError("Duplicate feature columns")
    result.attrs.update(
        source_frequency=str(step),
        windows=windows,
        groups=tuple(sorted(selected)),
        gap_policy=gap_policy,
        timestamp_semantics="bar_open",
        available_at="index + source_frequency",
    )
    return result


def build_features(
    bars: pd.DataFrame,
    frequency: str | pd.Timedelta,
    *,
    windows: tuple[int, ...] = (5, 20, 60),
    groups: Iterable[str] | None = None,
    names: Iterable[str] | None = None,
    source_frequency: str | pd.Timedelta = "1min",
    gap_policy: str = "reset",
    **aggregation_options,
) -> pd.DataFrame:
    """Combine enriched target bars with OHLCV features at the target frequency.

    Feature windows apply to TARGET bars; path/profile features use SOURCE bars.
    To propagate a completed higher-timeframe feature to source rows, join using
    ``available_at`` and source bar CLOSE times, never target bar-open timestamps.
    """
    if names is not None:
        if groups is not None:
            raise ValueError("Choose groups or exact names, not both")
        from pfe.features.targeted import build_selected_features

        return build_selected_features(
            bars,
            names,
            frequency,
            source_frequency=source_frequency,
            gap_policy=gap_policy,
            **aggregation_options,
        )
    if aggregation_options.get("partial", "drop") != "drop":
        raise ValueError("build_features requires partial='drop' to exclude provisional bars")
    enriched = enrich_bars(
        bars, frequency, source_frequency=source_frequency, **aggregation_options
    )
    if enriched.empty:
        schema = compute_features(
            normalize_bars(bars, frequency="1ns").iloc[:1],
            windows=windows,
            groups=groups,
            source_frequency=frequency,
            gap_policy=gap_policy,
            session_timezone=aggregation_options.get("session_timezone", "America/New_York"),
        )
        return enriched.join(schema.iloc[:0])
    standard = compute_features(
        enriched,
        windows=windows,
        groups=groups,
        source_frequency=frequency,
        gap_policy=gap_policy,
        session_timezone=aggregation_options.get("session_timezone", "America/New_York"),
    )
    result = enriched.join(standard)
    result.attrs.update(enriched.attrs)
    return result


def feature_catalog(features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Describe actual output columns, or list supported TA-Lib functions/groups.

    Pass a computed frame for its exact inventory (including configured bins,
    segments, windows). No claim of a finite catalog of arbitrary transforms.
    """
    if features is not None:
        records = []
        for name in features:
            family = name.split(".", 1)[0] if "." in name else "bars"
            records.append(
                {
                    "name": name,
                    "family": family,
                    "dtype": str(features[name].dtype),
                    "availability": (
                        "explicit decision_at"
                        if features.attrs.get("timestamp_semantics") == "explicit_decision"
                        else "explicit available_at"
                        if features.attrs.get("timestamp_semantics")
                        in ("exchange_session", "session_open")
                        else "bar_close"
                    ),
                    "description": name.replace("_", " "),
                }
            )
        return pd.DataFrame(records)
    records = [
        {"name": group, "family": group, "description": "Native feature family"}
        for group in FEATURE_GROUPS
    ]
    records.extend(
        {"name": group, "family": group, "description": "Enriched bar feature family"}
        for group in ("path", "profile", "quality")
    )
    from pfe.features.linear import LINEAR_FEATURE_GROUPS
    from pfe.features.macro import MACRO_FEATURE_GROUPS
    from pfe.features.options import OPTION_FEATURE_GROUPS
    from pfe.features.reference import REFERENCE_FEATURE_GROUPS

    records.extend(
        {"name": f"linear.{group}", "family": "linear", "description": "Walk-forward model outputs"}
        for group in LINEAR_FEATURE_GROUPS
    )
    records.extend(
        {
            "name": f"macro.{group}",
            "family": "macro",
            "description": "Availability-aware macro inputs",
        }
        for group in MACRO_FEATURE_GROUPS
    )
    records.extend(
        {"name": f"options.{group}", "family": "options", "description": "Observed option inputs"}
        for group in OPTION_FEATURE_GROUPS
    )
    records.extend(
        {
            "name": f"reference.{group}",
            "family": "reference",
            "description": "Known reference periods and explicitly matched market horizons",
        }
        for group in REFERENCE_FEATURE_GROUPS
    )
    records.append(
        {
            "name": "fundamentals",
            "family": "fundamentals",
            "description": "Known reported actuals with compatible accounting periods and units",
        }
    )
    from pfe.features.events import EVENT_FEATURE_GROUPS
    from pfe.features.graph import GRAPH_FEATURE_GROUPS
    from pfe.features.international import INTERNATIONAL_FEATURE_GROUPS

    for family, groups in (
        ("graph", GRAPH_FEATURE_GROUPS),
        ("events", EVENT_FEATURE_GROUPS),
        ("international", INTERNATIONAL_FEATURE_GROUPS),
    ):
        records.extend(
            {
                "name": f"{family}.{group}",
                "family": family,
                "description": "Sourced history at explicit decisions, with coverage masks",
            }
            for group in groups
        )
    records.append(
        {
            "name": "statements",
            "family": "statements",
            "description": "Reviewed textual actuals/guidance/estimates in original units",
        }
    )
    records.extend(
        [
            {
                "name": "cot",
                "family": "cot",
                "description": "Known COT release positioning in compatible reports",
            },
            {
                "name": "estimates",
                "family": "estimates",
                "description": "Licensed comparable estimates and pre-announcement surprises",
            },
        ]
    )
    for family, names in talib.get_function_groups().items():
        if family in {"Math Operators", "Math Transform"}:
            continue
        for name in names:
            fn = abstract.Function(name)
            records.append(
                {
                    "name": name,
                    "family": "patterns" if family == "Pattern Recognition" else "technical",
                    "description": fn.info["display_name"],
                    "inputs": str(dict(fn.input_names)),
                }
            )
    return pd.DataFrame(records)


def multi_timeframe_features(
    bars: pd.DataFrame,
    frequencies: tuple[str, ...],
    *,
    windows: tuple[int, ...] = (5, 20, 60),
    groups: Iterable[str] | None = None,
    names: Iterable[str] | None = None,
    source_frequency: str | pd.Timedelta = "1min",
    gap_policy: str = "reset",
    **aggregation_options,
) -> pd.DataFrame:
    """Align completed target features to source-bar CLOSE decision timestamps.

    Returned index retains source opens. A target value first appears at a source
    row whose close is >= target available_at, and never earlier. Last known
    target values persist across source gaps; age_seconds exposes their age.
    No backfill, interpolation or current unclosed-target values are used.
    """
    if isinstance(frequencies, str) or not frequencies:
        raise ValueError("frequencies must be a non-empty tuple of durations")
    durations = [fixed_duration(f, "frequency") for f in frequencies]
    if len(set(durations)) != len(durations):
        raise ValueError("frequencies must be unique durations")
    step = fixed_duration(source_frequency, "source_frequency")
    if names is not None:
        from pfe.features.selected import _validate

        data, _ = _validate(bars, step)
    else:
        data = normalize_bars(bars, frequency="1ns")
    left = pd.DataFrame({"decision_at": data.index + step})
    pieces = []
    for frequency in frequencies:
        target = build_features(
            data,
            frequency,
            source_frequency=source_frequency,
            windows=windows,
            groups=groups,
            names=names,
            gap_policy=gap_policy,
            **aggregation_options,
        )
        prefix = f"tf[{frequency}]"
        numeric = list(target.select_dtypes(include="number").columns)
        if target.empty:
            empty = pd.DataFrame({f"{prefix}.{name}": np.nan for name in numeric}, index=data.index)
            empty[f"{prefix}.age_seconds"] = np.nan
            pieces.append(empty)
            continue
        right = target[["available_at", *numeric]].reset_index(drop=True)
        aligned = pd.merge_asof(
            left,
            right,
            left_on="decision_at",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        values = aligned[numeric].copy()
        values.columns = [f"{prefix}.{name}" for name in numeric]
        values[f"{prefix}.age_seconds"] = (
            aligned.decision_at - aligned.available_at
        ).dt.total_seconds()
        values.index = data.index
        pieces.append(values)
    result = pd.concat(pieces, axis=1)
    result.attrs.update(
        timestamp_semantics="bar_open",
        available_at="index + source_frequency",
        source_frequency=str(step),
        frequencies=tuple(frequencies),
    )
    return result
