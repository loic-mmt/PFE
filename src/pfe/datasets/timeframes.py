"""Local OHLCV -> multi-timeframe observations -> framework-free model datasets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfe.data.bars import OHLCV, normalize_bars
from pfe.data.contracts import OBSERVATION_COLUMNS, _instant, normalize_observations
from pfe.datasets.core import (
    DatasetLimits,
    FeatureRegistry,
    build_feature_panel,
    build_sequence_dataset,
    forward_return_labels,
)


@dataclass(frozen=True)
class TimeframeSpec:
    frequency: str
    names: tuple[str, ...]
    units: dict[str, str]
    mode: str = "fixed"
    origin: str = "start_day"
    offset: str | None = None
    gap_policy: str = "reset"

    def __post_init__(self):
        from pfe.features.aggregation import fixed_duration

        if self.mode not in ("fixed", "session") or self.gap_policy not in ("reset", "allow"):
            raise ValueError("Choose fixed/session timeframe and reset/allow gap policy")
        if not isinstance(self.frequency, str) or not self.frequency.strip():
            raise ValueError("Timeframe needs an explicit frequency label")
        if self.mode == "fixed":
            fixed_duration(self.frequency, "frequency")
        if isinstance(self.names, str) or not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Timeframe needs distinct selected feature names")
        if (
            not isinstance(self.units, dict)
            or set(self.units) != set(self.names)
            or any(not isinstance(v, str) or not v or v != v.strip() for v in self.units.values())
        ):
            raise ValueError("Declare one exact unit per selected feature")
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "units", dict(self.units))

    def to_dict(self):
        return dict(
            frequency=self.frequency,
            names=list(self.names),
            units=self.units,
            mode=self.mode,
            origin=self.origin,
            offset=self.offset,
            gap_policy=self.gap_policy,
        )


@dataclass(frozen=True)
class TimeframeObservations:
    observations: pd.DataFrame
    registry: FeatureRegistry
    configuration: dict


@dataclass(frozen=True)
class TimeframeDataset:
    panel: object
    sequences: object
    labels: object
    features: TimeframeObservations


def _source(frame, step, policy):
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(
            "Timeframe bars require a DatetimeIndex; normalize source timestamps first"
        )
    bars = normalize_bars(frame, frequency="1ns")
    if policy == "explicit":
        if "available_at" not in frame:
            raise ValueError(
                "Provide source available_at or explicitly choose bar_close reconstruction"
            )
        known = pd.Series(
            [_instant(v, "source_available_at") for v in frame.available_at], index=frame.index
        ).reindex(bars.index)
    elif policy == "bar_close":
        known = pd.Series(bars.index + step, index=bars.index)
    else:
        raise ValueError("availability_policy must be explicit or bar_close")
    if (known < bars.index + step).any():
        raise ValueError("Source availability cannot precede its bar close")
    return bars, known


def _fixed(bars, available, spec, step):
    from pfe.features.selected import compute_selected_features
    from pfe.features.targeted import aggregate_selected_features

    target = pd.Timedelta(spec.frequency)
    if target < step or target.value % step.value:
        raise ValueError("Target timeframe must be an integer multiple of source_frequency")
    intrabar = [name for name in spec.names if name.startswith(("path.", "profile.", "quality."))]
    regular = [name for name in spec.names if name not in OHLCV and name not in intrabar]
    selected = tuple(dict.fromkeys([*OHLCV, "quality.complete", *intrabar]))
    base = aggregate_selected_features(
        bars,
        selected,
        spec.frequency,
        source_frequency=step,
        origin=spec.origin,
        offset=spec.offset,
        partial="drop",
    )
    complete = base["quality.complete"].eq(1)
    values = base.reindex(columns=spec.names).copy()
    if regular and complete.any():
        indicators = compute_selected_features(
            base.loc[complete, list(OHLCV)],
            regular,
            source_frequency=target,
            gap_policy=spec.gap_policy,
        )
        values.loc[:, regular] = indicators.reindex(base.index)
    values.loc[~complete, :] = np.nan
    closes = base.index + target
    # Recursive indicators can depend on the whole chronological prefix. This
    # conservative gate includes every source receipt in that prefix; raw prices
    # only need the source bars of their target bin.
    prefix = available.cummax()
    gates = []
    for start, close in zip(base.index, closes, strict=True):
        prefix_known = prefix.loc[prefix.index + step <= close]
        local = available.loc[(available.index >= start) & (available.index + step <= close)]
        gates.append(
            {
                name: max(
                    close,
                    local.max() if name in OHLCV or name in intrabar else prefix_known.iloc[-1],
                )
                for name in spec.names
            }
        )
    return values, closes, pd.DataFrame(gates, index=base.index), complete.to_numpy()


def _sessions(bars, available, spec, step, schedule, metadata, entity, source):
    from pfe.data.daily import aggregate_session_bars
    from pfe.features.daily import compute_daily_features

    if any(name.startswith(("path.", "profile.", "quality.")) for name in spec.names):
        raise ValueError(
            "Session datasets use daily indicators/OHLCV; intrabar paths need fixed bins"
        )
    if schedule is None or not isinstance(metadata, dict) or "instrument_id" not in metadata:
        raise ValueError("Session timeframe requires its exchange schedule and instrument metadata")
    daily = aggregate_session_bars(
        bars.assign(available_at=available),
        schedule=schedule,
        source_frequency=step,
        entity_id=entity,
        source_id=source,
        availability_basis="unknown",
        partial="keep",
        **metadata,
    )
    regular = [name for name in spec.names if name not in OHLCV]
    calculated = compute_daily_features(
        daily, schedule=schedule, names=regular or ("returns.simple",), gap_policy=spec.gap_policy
    )
    values = calculated.reindex(columns=spec.names)
    closes = pd.DatetimeIndex(calculated.session_close)
    gates = pd.DataFrame(
        {
            name: calculated.bar_available_at if name in OHLCV else calculated.available_at
            for name in spec.names
        },
        index=calculated.index,
    )
    return values, closes, gates, calculated["quality.feature_eligible"].to_numpy()


def build_timeframe_observations(
    bars_by_entity,
    timeframes,
    *,
    source_frequency,
    source_id,
    revision_id,
    availability_policy="explicit",
    schedules=None,
    session_metadata=None,
    source_documents=None,
    limits=None,
):
    """Compute only requested columns; no network, model or implicit timeframe.

    Fixed durations are explicit (including a 24h bin for continuous markets).
    Equity daily sessions require mode='session' and the exchange schedule.
    Source receipt clocks are mandatory by default; bar_close is an explicit
    non-PIT historical reconstruction. Inputs must already resolve source-bar
    revisions; conflicting timestamps are rejected by normalize_bars.
    """
    from pfe.features.aggregation import fixed_duration

    limits = limits or DatasetLimits()
    step = fixed_duration(source_frequency, "source_frequency")
    specs = tuple(TimeframeSpec(**v) if isinstance(v, dict) else v for v in timeframes)
    if not specs or any(not isinstance(v, TimeframeSpec) for v in specs):
        raise ValueError("Supply timeframe specifications")
    if len({s.frequency for s in specs}) != len(specs):
        raise ValueError("Timeframe labels must be unique")
    if (
        not isinstance(bars_by_entity, dict)
        or not bars_by_entity
        or len(bars_by_entity) > limits.max_entities
    ):
        raise ValueError("Supply a bounded explicit entity -> bars mapping")
    if any(
        not isinstance(v, str) or not v.strip() for v in (source_id, revision_id, *bars_by_entity)
    ):
        raise ValueError("Source, revision and entity IDs must be explicit")
    registry = FeatureRegistry(
        [
            dict(
                name=f"tf[{spec.frequency}].{name}",
                metric=f"tf[{spec.frequency}].{name}",
                unit=spec.units[name],
                source_id=source_id,
                frequency=spec.frequency,
                window=f"{name}; target={spec.frequency}; gap={spec.gap_policy}",
                dependencies=tuple(OHLCV),
            )
            for spec in specs
            for name in spec.names
        ]
    )
    if len(registry.features) > limits.max_features:
        raise ValueError("Timeframe registry exceeds feature budget")
    rows = []
    for entity, frame in bars_by_entity.items():
        if (
            len(frame) > limits.max_times
            or frame.memory_usage(index=True, deep=True).sum() > limits.max_bytes
        ):
            raise ValueError("Source bars exceed timeframe budgets")
        bars, available = _source(frame, step, availability_policy)
        for spec in specs:
            if spec.mode == "fixed":
                values, closes, gates, eligible = _fixed(bars, available, spec, step)
            else:
                values, closes, gates, eligible = _sessions(
                    bars,
                    available,
                    spec,
                    step,
                    (schedules or {}).get(entity),
                    (session_metadata or {}).get(entity),
                    entity,
                    source_id,
                )
            for i in range(len(values)):
                for name in spec.names:
                    value = values.iloc[i][name]
                    rows.append(
                        dict(
                            entity_id=entity,
                            observation_date=closes[i].isoformat(),
                            published_at=None,
                            available_at=gates.iloc[i][name],
                            value=float(value) if pd.notna(value) else None,
                            metric=f"tf[{spec.frequency}].{name}",
                            unit=spec.units[name],
                            source_id=source_id,
                            revision_id=revision_id,
                            schema_version="1",
                            frequency=spec.frequency,
                            availability_basis="reconstructed"
                            if availability_policy == "bar_close"
                            else "unknown",
                            point_in_time=False,
                            source_document_id=(source_documents or {}).get(entity),
                            **{"quality.feature_eligible": bool(eligible[i])},
                        )
                    )
                    if len(rows) * 512 > limits.max_bytes:
                        raise ValueError("Timeframe observations exceed byte budget")
    frame = (
        normalize_observations(pd.DataFrame(rows))
        if rows
        else normalize_observations(pd.DataFrame(columns=OBSERVATION_COLUMNS))
    )
    return TimeframeObservations(
        frame,
        registry,
        dict(
            timeframes=[s.to_dict() for s in specs],
            source_frequency=str(step),
            availability_policy=availability_policy,
            revision_id=revision_id,
            source_id=source_id,
        ),
    )


def build_timeframe_dataset(
    bars_by_entity,
    timeframes,
    *,
    decisions,
    lookback,
    source_frequency,
    source_id,
    revision_id,
    label_feature=None,
    label_horizon=1,
    stride=1,
    limits=None,
    **options,
):
    """Align completed timeframes to any explicit intraday/session decision grid.

    Lookback, stride and label_horizon count that grid's steps, not days. Higher
    timeframe values persist only after their known time; per-cell clocks/masks
    retain age/provenance. Labels use the explicitly selected price feature.
    """
    features = build_timeframe_observations(
        bars_by_entity,
        timeframes,
        source_frequency=source_frequency,
        source_id=source_id,
        revision_id=revision_id,
        limits=limits,
        **options,
    )
    panel = build_feature_panel(
        decisions,
        features.observations,
        entity_ids=list(bars_by_entity),
        registry=features.registry,
        limits=limits,
    )
    labels = (
        forward_return_labels(panel, price_feature=label_feature, horizon=label_horizon)
        if label_feature is not None
        else None
    )
    sequences = build_sequence_dataset(
        panel, lookback=lookback, labels=labels, stride=stride, limits=limits
    )
    return TimeframeDataset(panel, sequences, labels, features)
