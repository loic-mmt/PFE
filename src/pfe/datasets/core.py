"""Auditable numerical panels and bounded GRU/TCN windows, without DL frameworks."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant, _missing, normalize_observations, observations_asof
from pfe.features.graph import decision_grid

NAT = np.iinfo(np.int64).min


@dataclass(frozen=True)
class DatasetLimits:
    max_bytes: int = 8_388_608
    max_samples: int = 4096
    max_times: int = 4096
    max_entities: int = 128
    max_features: int = 256
    max_edges: int = 8192
    max_events: int = 32

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Dataset budgets must be positive integers")


def check_bytes(limits, *arrays, extra=0):
    if sum(a.nbytes for a in arrays) + extra > limits.max_bytes:
        raise ValueError("Dataset exceeds max_bytes")


def digest_metadata(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    metric: str
    unit: str
    source_id: str
    window: str = "point"
    dependencies: tuple[str, ...] = ()
    max_age: str | None = None
    frequency: str | None = None
    instrument_id: str | None = None

    def __post_init__(self):
        for label in ("name", "metric", "unit", "source_id", "window"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("Feature names/metrics/units/sources/windows must be explicit")
        if not isinstance(self.dependencies, (list, tuple)) or any(
            not isinstance(v, str) or not v for v in self.dependencies
        ):
            raise ValueError("Feature dependencies require explicit input names")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if self.instrument_id is not None and (
            not isinstance(self.instrument_id, str)
            or not self.instrument_id
            or self.instrument_id != self.instrument_id.strip()
        ):
            raise ValueError("Feature instrument_id must be an explicit identity or null")
        if self.max_age is not None and (
            pd.isna(pd.Timedelta(self.max_age)) or pd.Timedelta(self.max_age) < pd.Timedelta(0)
        ):
            raise ValueError("Feature max_age must be nonnegative")


class FeatureRegistry:
    def __init__(self, features):
        self.features = tuple(FeatureSpec(**s) if isinstance(s, dict) else s for s in features)
        if (
            not self.features
            or any(not isinstance(s, FeatureSpec) for s in self.features)
            or len({s.name for s in self.features}) != len(self.features)
        ):
            raise ValueError("Feature registry requires distinct named specifications")

    def to_dict(self):
        return {"schema_version": "1", "features": [asdict(s) for s in self.features]}

    @property
    def fingerprint(self):
        return digest_metadata(self.to_dict())


@dataclass(frozen=True)
class FeaturePanel:
    values: np.ndarray
    mask: np.ndarray
    available_at_ns: np.ndarray
    observation_at_ns: np.ndarray
    provenance_codes: np.ndarray
    decisions_ns: np.ndarray
    entity_ids: tuple[str, ...]
    registry: FeatureRegistry
    provenance: tuple[dict, ...]

    @property
    def fingerprint(self):
        state = hashlib.sha256(
            digest_metadata(
                {
                    "registry": self.registry.to_dict(),
                    "entities": self.entity_ids,
                    "provenance": self.provenance,
                }
            ).encode()
        )
        for array in (
            self.values,
            self.mask,
            self.available_at_ns,
            self.observation_at_ns,
            self.provenance_codes,
            self.decisions_ns,
        ):
            state.update(str((array.shape, array.dtype.str)).encode())
            state.update(np.ascontiguousarray(array).tobytes())
        return state.hexdigest()


def build_feature_panel(decisions, observations, *, entity_ids, registry, limits=None):
    """Align selected numeric vintages with per-cell source and time provenance.

    Only economic observations and availability at/before the decision are used.
    Forecast-period facts must first become decision-dated derived features.
    Missing/latest-null/stale values remain masked; no cross-source fallback.
    """
    limits = limits or DatasetLimits()
    registry = registry if isinstance(registry, FeatureRegistry) else FeatureRegistry(registry)
    index = decision_grid(decisions, entity_ids)
    t, n, f = len(index), len(entity_ids), len(registry.features)
    if t > limits.max_times or n > limits.max_entities or f > limits.max_features:
        raise ValueError("Panel axes exceed dataset budgets")
    shape = (t, n, f)
    if t * n * f * 29 + t * 8 > limits.max_bytes:
        raise ValueError("Panel allocation exceeds max_bytes")
    frame = normalize_observations(observations)
    if "quality.feature_eligible" in frame:
        flags = frame["quality.feature_eligible"]
        if not flags.map(lambda v: _missing(v) or isinstance(v, (bool, np.bool_))).all():
            raise ValueError("quality.feature_eligible requires booleans or missing")
        frame["quality.feature_eligible"] = flags.map(lambda v: True if _missing(v) else bool(v))
    values = np.full(shape, np.nan)
    mask = np.zeros(shape, dtype=bool)
    available = np.full(shape, NAT, dtype=np.int64)
    observed = np.full(shape, NAT, dtype=np.int64)
    codes = np.full(shape, -1, dtype=np.int32)
    provenance, lookup = [], {}
    for k, spec in enumerate(registry.features):
        selected = frame.loc[
            frame.source_id.eq(spec.source_id)
            & frame.metric.eq(spec.metric)
            & frame.unit.eq(spec.unit)
            & frame.entity_id.isin(entity_ids)
        ]
        if spec.instrument_id is not None:
            selected = selected.loc[selected.instrument_id.eq(spec.instrument_id)]
        elif selected.groupby("entity_id").instrument_id.nunique(dropna=False).gt(1).any():
            raise ValueError("Select instrument_id for features with ambiguous instruments")
        for i, decision in enumerate(index):
            known = observations_asof(selected, decision)
            economic = pd.to_datetime(known.observation_date, utc=True, format="mixed")
            known = known.loc[economic <= decision]
            for j, entity in enumerate(entity_ids):
                points = known.loc[known.entity_id.eq(entity)]
                if points.empty:
                    continue
                point = points.loc[
                    pd.to_datetime(points.observation_date, utc=True, format="mixed").idxmax()
                ]
                at, obs = (
                    point.available_at.value,
                    pd.Timestamp(point.observation_date, tz="UTC").value
                    if len(point.observation_date) == 10
                    else _instant(point.observation_date, "observation_date").value,
                )
                available[i, j, k], observed[i, j, k] = at, obs
                proof = {
                    key: point.get(key)
                    for key in (
                        "entity_id",
                        "instrument_id",
                        "metric",
                        "unit",
                        "source_id",
                        "revision_id",
                        "source_document_id",
                        "availability_basis",
                        "observation_date",
                    )
                }
                proof["point_in_time"] = bool(point.point_in_time)
                identity = digest_metadata(proof)
                if identity not in lookup:
                    lookup[identity] = len(provenance)
                    provenance.append(proof)
                codes[i, j, k] = lookup[identity]
                value = point.value
                if value is not None and (
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                ):
                    raise ValueError(
                        "Dataset features must be numeric; derive structured facts first"
                    )
                fresh = (
                    spec.max_age is None or decision.value - at <= pd.Timedelta(spec.max_age).value
                )
                compatible = spec.frequency is None or point.get("frequency") == spec.frequency
                if (
                    value is not None
                    and fresh
                    and compatible
                    and point.get("quality.feature_eligible", True)
                ):
                    values[i, j, k], mask[i, j, k] = value, True
    decision_ns = index.as_unit("ns").asi8
    check_bytes(
        limits,
        values,
        mask,
        available,
        observed,
        codes,
        decision_ns,
        extra=len(json.dumps(provenance).encode()),
    )
    return FeaturePanel(
        values,
        mask,
        available,
        observed,
        codes,
        decision_ns,
        tuple(entity_ids),
        registry,
        tuple(provenance),
    )


def feature_frame_to_observations(
    frame, registry, *, available_at, revision_id, source_document_id=None, point_in_time=False
):
    """Bridge computed (decision_at,entity_id) features with explicit known times.

    ``available_at`` is a matching DataFrame, never inferred from row dates.
    Caller must retain upstream latency/review clocks when constructing it.
    """
    registry = registry if isinstance(registry, FeatureRegistry) else FeatureRegistry(registry)
    if (
        not isinstance(frame.index, pd.MultiIndex)
        or frame.index.names != ["decision_at", "entity_id"]
        or not frame.index.is_unique
    ):
        raise ValueError("Computed features need unique (decision_at,entity_id) index")
    names = [s.name for s in registry.features]
    if (
        not available_at.index.equals(frame.index)
        or not set(names) <= set(available_at.columns)
        or not set(names) <= set(frame.columns)
    ):
        raise ValueError("Computed features require matching per-cell availability")
    records = []
    for (decision, entity), row in frame.iterrows():
        for spec in registry.features:
            stamp = available_at.loc[(decision, entity), spec.name]
            if pd.isna(stamp):
                if pd.notna(row[spec.name]):
                    raise ValueError("Known feature value needs availability")
                continue
            records.append(
                dict(
                    entity_id=entity,
                    instrument_id=spec.instrument_id,
                    observation_date=_instant(decision, "decision_at").isoformat(),
                    available_at=_instant(stamp, "available_at"),
                    published_at=None,
                    value=row[spec.name],
                    metric=spec.metric,
                    unit=spec.unit,
                    source_id=spec.source_id,
                    revision_id=revision_id,
                    source_document_id=source_document_id,
                    schema_version="1",
                    point_in_time=point_in_time,
                    availability_basis="unknown",
                )
            )
    from pfe.data.eia import _observations

    return normalize_observations(pd.DataFrame(records)) if records else _observations([])


@dataclass(frozen=True)
class ForwardLabels:
    values: np.ndarray
    mask: np.ndarray
    start_ns: np.ndarray
    end_ns: np.ndarray
    available_at_ns: np.ndarray
    panel_fingerprint: str
    horizon: int


def forward_return_labels(panel, *, price_feature, horizon=1):
    """Labels use future decision-grid prices; never appended to input features."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Forward horizon must be a positive number of grid steps")
    names = [s.name for s in panel.registry.features]
    k = names.index(price_feature)
    t, n, _ = panel.values.shape
    values = np.full((t, n), np.nan)
    mask = np.zeros((t, n), dtype=bool)
    starts = np.broadcast_to(panel.decisions_ns[:, None], (t, n)).copy()
    ends = np.full((t, n), NAT, dtype=np.int64)
    available = np.full((t, n), NAT, dtype=np.int64)
    if horizon < t:
        first, last = panel.values[:-horizon, :, k], panel.values[horizon:, :, k]
        eligible = (
            panel.mask[:-horizon, :, k]
            & panel.mask[horizon:, :, k]
            & (first > 0)
            & (last > 0)
            & (panel.observation_at_ns[horizon:, :, k] > panel.observation_at_ns[:-horizon, :, k])
        )
        values[:-horizon] = np.where(eligible, last / np.where(first > 0, first, 1) - 1, np.nan)
        mask[:-horizon] = eligible
        ends[:-horizon] = panel.decisions_ns[horizon:, None]
        available[:-horizon] = np.maximum(ends[:-horizon], panel.available_at_ns[horizon:, :, k])
    return ForwardLabels(values, mask, starts, ends, available, panel.fingerprint, horizon)


@dataclass(frozen=True)
class SequenceDataset:
    x: np.ndarray
    mask: np.ndarray
    end_indices: np.ndarray
    entity_indices: np.ndarray
    decision_ns: np.ndarray
    window_start_ns: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    label_end_ns: np.ndarray
    label_available_at_ns: np.ndarray
    panel_fingerprint: str
    registry_fingerprint: str
    lookback: int

    @property
    def fingerprint(self):
        state = hashlib.sha256(
            digest_metadata(
                {
                    "panel": self.panel_fingerprint,
                    "registry": self.registry_fingerprint,
                    "lookback": self.lookback,
                }
            ).encode()
        )
        for array in (
            self.x,
            self.mask,
            self.end_indices,
            self.entity_indices,
            self.decision_ns,
            self.window_start_ns,
            self.y,
            self.y_mask,
            self.label_end_ns,
            self.label_available_at_ns,
        ):
            state.update(str((array.shape, array.dtype.str)).encode())
            state.update(np.ascontiguousarray(array).tobytes())
        return state.hexdigest()


def iter_sequence_batches(panel, *, lookback, labels=None, stride=1, batch_size=128, limits=None):
    limits = limits or DatasetLimits()
    if any(type(v) is not int or v < 1 for v in (lookback, stride, batch_size)):
        raise ValueError("Window/stride/batch_size must be positive integers")
    t, n, f = panel.values.shape
    if lookback > t:
        raise ValueError("Lookback exceeds panel history")
    if labels is not None and labels.panel_fingerprint != panel.fingerprint:
        raise ValueError("Labels belong to another panel")
    total = len(range(lookback - 1, t, stride)) * n
    if total > limits.max_samples:
        raise ValueError("Sequence sample count exceeds budget")
    sample_ends = np.repeat(np.arange(lookback - 1, t, stride), n)
    sample_entities = np.tile(np.arange(n), total // n)
    fingerprint = panel.fingerprint
    for offset in range(0, total, batch_size):
        ends, entities = (
            sample_ends[offset : offset + batch_size],
            sample_entities[offset : offset + batch_size],
        )
        if len(ends) * lookback * f * 9 > limits.max_bytes:
            raise ValueError("Sequence batch allocation exceeds max_bytes")
        times = ends[:, None] - np.arange(lookback - 1, -1, -1)[None, :]
        mask = panel.mask[times, entities[:, None], :]
        x = np.where(mask, panel.values[times, entities[:, None], :], 0.0)
        y = labels.values[ends, entities] if labels is not None else np.full(len(ends), np.nan)
        ym = labels.mask[ends, entities] if labels is not None else np.zeros(len(ends), dtype=bool)
        le = labels.end_ns[ends, entities] if labels is not None else np.full(len(ends), NAT)
        la = (
            labels.available_at_ns[ends, entities]
            if labels is not None
            else np.full(len(ends), NAT)
        )
        batch = SequenceDataset(
            x,
            mask,
            ends,
            entities,
            panel.decisions_ns[ends],
            panel.decisions_ns[ends - lookback + 1],
            y,
            ym,
            le,
            la,
            fingerprint,
            panel.registry.fingerprint,
            lookback,
        )
        check_bytes(
            limits,
            *(
                getattr(batch, key)
                for key in (
                    "x",
                    "mask",
                    "end_indices",
                    "entity_indices",
                    "decision_ns",
                    "window_start_ns",
                    "y",
                    "y_mask",
                    "label_end_ns",
                    "label_available_at_ns",
                )
            ),
        )
        yield batch


def build_sequence_dataset(panel, *, lookback, labels=None, stride=1, limits=None):
    limits = limits or DatasetLimits()
    t, n, f = panel.values.shape
    if type(lookback) is not int or not 1 <= lookback <= t or type(stride) is not int or stride < 1:
        raise ValueError("Require positive stride and lookback within panel history")
    samples = len(range(lookback - 1, t, stride)) * n
    if samples * (lookback * f * 9 + 65) > limits.max_bytes:
        raise ValueError("Sequence allocation exceeds max_bytes; use iter_sequence_batches")
    batches = list(
        iter_sequence_batches(panel, lookback=lookback, labels=labels, stride=stride, limits=limits)
    )
    arrays = [
        np.concatenate([getattr(b, name) for b in batches], axis=0)
        for name in (
            "x",
            "mask",
            "end_indices",
            "entity_indices",
            "decision_ns",
            "window_start_ns",
            "y",
            "y_mask",
            "label_end_ns",
            "label_available_at_ns",
        )
    ]
    check_bytes(limits, *arrays)
    return SequenceDataset(*arrays, panel.fingerprint, panel.registry.fingerprint, lookback)
