"""Streaming canonical observations, window overlap and immutable dataset shards."""

import hashlib
import io
import json
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant, normalize_observations, observations_asof
from pfe.datasets.core import (
    DatasetLimits,
    FeatureRegistry,
    build_feature_panel,
    build_sequence_dataset,
    forward_return_labels,
)
from pfe.datasets.io import _load_npz, _publish, load_model_dataset, save_model_dataset
from pfe.datasets.temporal import TrainOnlyScaler


@dataclass(frozen=True)
class DatasetPartition:
    panel: object
    global_offset: int
    core_start: int
    core_stop: int
    sequences: object = None


def iter_parquet_observations(
    paths, *, batch_rows=256, max_batch_bytes=8_388_608, max_row_group_bytes=67_108_864
):
    """Read ordered Parquet batches; never load an entire historical panel.

    Files and row groups must already be ordered by available_at. Row group
    expanded-size metadata is checked before decoding. Canonical observations
    may be partitioned by receipt upstream; economic-date-only sorting is invalid.
    """
    import pyarrow.parquet as pq

    if any(type(v) is not int or v < 1 for v in (batch_rows, max_batch_bytes, max_row_group_bytes)):
        raise ValueError("Parquet streaming budgets must be positive")
    for path in paths:
        file = pq.ParquetFile(path)
        for group in range(file.num_row_groups):
            if file.metadata.row_group(group).total_byte_size > max_row_group_bytes:
                raise ValueError("Parquet row group exceeds decode budget; repartition upstream")
            for batch in file.iter_batches(batch_size=batch_rows, row_groups=[group]):
                if batch.nbytes > max_batch_bytes:
                    raise ValueError("Parquet observation batch exceeds byte budget")
                yield batch.to_pandas()


def _trim_state(frame, cutoff):
    known = observations_asof(frame, cutoff)
    economic = pd.to_datetime(known.observation_date, utc=True, format="mixed")
    past = known.loc[economic <= cutoff].copy()
    past["_economic"] = economic.loc[past.index]
    keys = ["entity_id", "metric", "source_id", "instrument_id", "unit"]
    past = (
        past.sort_values("_economic", kind="stable")
        .drop_duplicates(keys, keep="last")
        .drop(columns="_economic")
    )
    # Known forecasts can become eligible at a later economic date. Unseen
    # revisions and old-period updates are retained until their knowledge gate.
    return pd.concat(
        [past, known.loc[economic > cutoff], frame.loc[frame.available_at > cutoff]],
        ignore_index=True,
    )


def iter_panel_partitions(
    decisions,
    observation_batches,
    *,
    entity_ids,
    registry,
    partition_size=128,
    lookback=1,
    lookahead=0,
    limits=None,
):
    """Bound state to latest points plus window context and known future periods.

    Input batches must be disjoint canonical rows sorted by available_at. Late
    revisions are valid even for old economic periods. The explicit decision
    axis is held as timestamp metadata; values/proofs are built per partition.
    """
    limits = limits or DatasetLimits()
    if (
        any(type(v) is not int or v < 1 for v in (partition_size, lookback))
        or type(lookahead) is not int
        or lookahead < 0
    ):
        raise ValueError("Partition/lookback must be positive; lookahead nonnegative")
    index = pd.DatetimeIndex(decisions)
    if (
        index.tz is None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Partition decisions must be increasing unique aware timestamps")
    if not len(index):
        raise ValueError("Partition decisions must not be empty")
    if partition_size + lookback - 1 + lookahead > limits.max_times:
        raise ValueError("Partition plus context exceeds time-axis budget")
    registry = registry if isinstance(registry, FeatureRegistry) else FeatureRegistry(registry)
    iterator = iter(observation_batches)
    pending, state = None, None
    last_input = None
    exhausted = False
    for start in range(0, len(index), partition_size):
        stop = min(start + partition_size, len(index))
        left = max(0, start - lookback + 1)
        right = min(len(index), stop + lookahead)
        through = index[right - 1]
        while True:
            if pending is None:
                try:
                    batch = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                if (
                    not isinstance(batch, pd.DataFrame)
                    or batch.memory_usage(index=True, deep=True).sum() > limits.max_bytes
                ):
                    raise ValueError("Observation batch exceeds byte budget")
                pending = normalize_observations(batch)
                if pending.empty:
                    pending = None
                    continue
                if last_input is not None and pending.available_at.min() < last_input:
                    raise ValueError("Observation batches must be sorted by available_at")
                last_input = pending.available_at.max()
            ready = pending.loc[pending.available_at <= through]
            pending = pending.loc[pending.available_at > through]
            selected = ready.entity_id.isin(entity_ids) & False
            for spec in registry.features:
                match = (
                    ready.entity_id.isin(entity_ids)
                    & ready.metric.eq(spec.metric)
                    & ready.source_id.eq(spec.source_id)
                    & ready.unit.eq(spec.unit)
                )
                if spec.instrument_id is not None:
                    match &= ready.instrument_id.eq(spec.instrument_id)
                selected |= match
            ready = ready.loc[selected]
            if not ready.empty:
                state = (
                    ready
                    if state is None or state.empty
                    else pd.concat([state, ready], ignore_index=True)
                )
                if state.memory_usage(index=True, deep=True).sum() > limits.max_bytes:
                    raise ValueError(
                        "Observation state exceeds byte budget; narrow registry/source batches"
                    )
            if len(pending):
                break
            pending = None
        if state is None:
            from pfe.data.contracts import OBSERVATION_COLUMNS

            state = pd.DataFrame(columns=OBSERVATION_COLUMNS)
        panel = build_feature_panel(
            index[left:right], state, entity_ids=entity_ids, registry=registry, limits=limits
        )
        yield DatasetPartition(panel, left, start, stop)
        if stop < len(index):
            cutoff = index[max(0, stop - lookback + 1)]
            state = _trim_state(state, cutoff)
        if exhausted:
            pending = None


def iter_partitioned_sequences(
    decisions,
    observation_batches,
    *,
    entity_ids,
    registry,
    lookback,
    label_feature=None,
    label_horizon=1,
    stride=1,
    partition_size=128,
    limits=None,
):
    """Same numerical windows/labels as full build, with local shard indices.

    global_offset + sequence.end_indices gives the full-panel index. Core decision
    ranges do not overlap, input lookback and label lookahead do. No per-shard fit.
    """
    if type(stride) is not int or stride < 1 or type(label_horizon) is not int or label_horizon < 1:
        raise ValueError("Stride and label horizon must be positive")
    for part in iter_panel_partitions(
        decisions,
        observation_batches,
        entity_ids=entity_ids,
        registry=registry,
        partition_size=partition_size,
        lookback=lookback,
        lookahead=label_horizon if label_feature is not None else 0,
        limits=limits,
    ):
        if len(part.panel.decisions_ns) < lookback:
            continue
        labels = (
            forward_return_labels(part.panel, price_feature=label_feature, horizon=label_horizon)
            if label_feature is not None
            else None
        )
        data = build_sequence_dataset(part.panel, lookback=lookback, labels=labels, limits=limits)
        global_ends = data.end_indices + part.global_offset
        selected = (
            (global_ends >= part.core_start)
            & (global_ends < part.core_stop)
            & ((global_ends - lookback + 1) % stride == 0)
        )
        arrays = {
            field.name: getattr(data, field.name)[selected]
            for field in fields(data)
            if isinstance(getattr(data, field.name), np.ndarray)
        }
        if selected.any():
            yield replace(part, sequences=replace(data, **arrays))


class _TrainingState:
    def __init__(self, train_end):
        self.end = _instant(train_end, "train_end").value
        self.registry = None
        self.entities = None
        self.mean = self.m2 = self.count = None
        self.last_cells = {}
        self.last_sample = None
        self.digest = hashlib.sha256(str(self.end).encode())

    def update(self, part):
        data = part.sequences
        if data is None:
            raise ValueError("Training requires sequence partitions with labels")
        if self.registry is None:
            self.registry = data.registry_fingerprint
            self.entities = part.panel.entity_ids
            self.mean = np.zeros(data.x.shape[-1])
            self.m2 = np.zeros_like(self.mean)
            self.count = np.zeros(len(self.mean), dtype=np.int64)
            self.digest.update(self.registry.encode())
        if data.registry_fingerprint != self.registry or part.panel.entity_ids != self.entities:
            raise ValueError("Partition feature registries differ")
        for i in range(len(data.x)):
            end = int(data.end_indices[i] + part.global_offset)
            entity = int(data.entity_indices[i])
            key = (end, entity)
            if self.last_sample is not None and key <= self.last_sample:
                raise ValueError("Sequence partitions must be disjoint and chronologically ordered")
            self.last_sample = key
            if not (
                data.y_mask[i]
                and data.decision_ns[i] < self.end
                and data.label_end_ns[i] < self.end
                and data.label_available_at_ns[i] < self.end
            ):
                continue
            for offset in range(data.lookback):
                cell = end - data.lookback + 1 + offset
                if cell <= self.last_cells.get(entity, -1):
                    continue
                observed = data.mask[i, offset] & np.isfinite(data.x[i, offset])
                value = data.x[i, offset]
                self.digest.update(np.array([cell, entity], dtype=np.int64).tobytes())
                self.digest.update(np.where(observed, value, 0).tobytes())
                self.digest.update(observed.tobytes())
                count = self.count + observed
                delta = value - self.mean
                self.mean += np.divide(delta, count, out=np.zeros_like(delta), where=observed)
                self.m2 += np.where(observed, delta * (value - self.mean), 0)
                self.count = count
            self.last_cells[entity] = end

    def finish(self):
        if self.count is None or not self.count.any():
            raise ValueError("No eligible training cells before purged boundary")
        scale = np.sqrt(
            np.divide(self.m2, self.count, out=np.zeros_like(self.mean), where=self.count > 0)
        )
        scale[scale == 0] = 1
        return PartitionedScaler(
            self.mean, scale, self.count, self.registry, self.digest.hexdigest(), self.end
        )


@dataclass(frozen=True)
class PartitionedScaler:
    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray
    registry_fingerprint: str
    training_fingerprint: str
    train_end_ns: int

    def transform(self, data):
        return TrainOnlyScaler(
            self.mean, self.scale, self.count, self.registry_fingerprint, self.training_fingerprint
        ).transform(data)

    def to_dict(self):
        return dict(
            mean=self.mean.tolist(),
            scale=self.scale.tolist(),
            count=self.count.tolist(),
            registry_fingerprint=self.registry_fingerprint,
            training_fingerprint=self.training_fingerprint,
            train_end_ns=self.train_end_ns,
            fit_scope="unique_purged_training_stream_cells",
        )


def fit_partitioned_scaler(partitions, *, train_end):
    state = _TrainingState(train_end)
    for part in partitions:
        state.update(part)
    return state.finish()


def save_partitioned_dataset(
    directory,
    partitions,
    *,
    scaler=None,
    configuration=None,
    limits=None,
    max_archive_bytes=8_388_608,
    max_partitions=4096,
    graph_history=None,
    graph_max_age=None,
):
    """Publish final index only after every immutable shard and scaler is verified.

    A caller can replay the same input iterator factory after interruption. Each
    shard is independently bounded; the explicit total archive cap includes raw
    shard files, optional standardized inputs and index metadata.
    """
    if any(type(v) is not int or v < 1 for v in (max_archive_bytes, max_partitions)):
        raise ValueError("Partition archive budgets must be positive")
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Partition directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    if sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) > max_archive_bytes:
        raise ValueError("Existing partitioned archive exceeds total byte budget")
    entries = []
    total = 0
    registry = entities = None
    training = (
        _TrainingState(pd.Timestamp(scaler.train_end_ns, tz="UTC")) if scaler is not None else None
    )
    for i, part in enumerate(partitions):
        if i >= max_partitions:
            raise ValueError("Partition count exceeds budget")
        if part.sequences is None and graph_history is None:
            raise ValueError("Save sequence partitions")
        if registry is None:
            registry, entities = part.panel.registry.to_dict(), part.panel.entity_ids
        if registry != part.panel.registry.to_dict() or entities != part.panel.entity_ids:
            raise ValueError("Partition registry/entity axes differ")
        name = f"part-{i:06d}"
        target = directory / name
        scaled_body = None
        remaining = max_archive_bytes - total
        if scaler is not None:
            if part.sequences is None:
                raise ValueError("Partitioned scaler requires sequence training inputs")
            training.update(part)
            values, mask = scaler.transform(part.sequences)
            stream = io.BytesIO()
            np.savez(stream, x=values, mask=mask)
            scaled_body = stream.getvalue()
            remaining -= len(scaled_body)
        if remaining < 1:
            raise ValueError("Partitioned archive exceeds total byte budget")
        shard_limits = replace(
            limits or DatasetLimits(),
            max_bytes=min((limits or DatasetLimits()).max_bytes, remaining),
        )
        graphs = ()
        if graph_history is not None:
            from pfe.datasets.structured import iter_graph_tensors

            graphs = (
                graph
                for j, graph in enumerate(
                    iter_graph_tensors(
                        part.panel, graph_history, max_age=graph_max_age, limits=shard_limits
                    )
                )
                if part.core_start <= j + part.global_offset < part.core_stop
            )
        save_model_dataset(
            target,
            part.panel,
            sequences=part.sequences,
            graphs=graphs,
            configuration={
                "global_offset": part.global_offset,
                "core_start": part.core_start,
                "core_stop": part.core_stop,
            },
            limits=shard_limits,
        )
        entry = dict(
            path=name,
            manifest_sha256=hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest(),
            global_offset=part.global_offset,
            core_start=part.core_start,
            core_stop=part.core_stop,
            samples=len(part.sequences.x) if part.sequences is not None else 0,
        )
        if scaled_body is not None:
            _publish(target / "scaled.npz", scaled_body)
            entry["scaled_sha256"] = hashlib.sha256(scaled_body).hexdigest()
        total += sum(path.stat().st_size for path in target.iterdir() if path.is_file())
        entries.append(entry)
        if total > max_archive_bytes:
            raise ValueError("Partitioned archive exceeds total byte budget")
    if not entries:
        raise ValueError("No sequence partitions to publish")
    expected_paths = {entry["path"] for entry in entries} | {"manifest.json"}
    if any(path.name not in expected_paths for path in directory.iterdir()):
        raise ValueError("Unexpected files or stale shards in partition output directory")
    if scaler is not None and training.finish().training_fingerprint != scaler.training_fingerprint:
        raise ValueError("Scaler belongs to a different purged training stream")
    manifest = dict(
        schema_version="1",
        complete=True,
        registry=registry,
        entity_ids=entities,
        shards=entries,
        configuration=configuration or {},
        scaler=scaler.to_dict() if scaler is not None else None,
    )
    body = json.dumps(manifest, sort_keys=True, allow_nan=False, indent=2).encode()
    if total + len(body) > max_archive_bytes:
        raise ValueError("Partitioned manifest exceeds total archive budget")
    _publish(directory / "manifest.json", body)
    return manifest


def iter_saved_partitions(directory, *, limits=None, max_manifest_bytes=1_048_576):
    """Verify and load one shard at a time; never materialize all sequence arrays."""
    if type(max_manifest_bytes) is not int or max_manifest_bytes < 1:
        raise ValueError("Manifest byte budget must be positive")
    directory = Path(directory)
    with (directory / "manifest.json").open("rb") as stream:
        body = stream.read(max_manifest_bytes + 1)
    if len(body) > max_manifest_bytes:
        raise ValueError("Partition manifest exceeds budget")
    manifest = json.loads(body)
    if manifest.get("complete") is not True or manifest.get("schema_version") != "1":
        raise ValueError("Incomplete partition manifest")
    for entry in manifest["shards"]:
        if not re.fullmatch(r"part-\d{6}", entry["path"]):
            raise ValueError("Invalid partition path")
        path = directory / entry["path"]
        with (path / "manifest.json").open("rb") as stream:
            shard_manifest = stream.read(max_manifest_bytes + 1)
        if (
            path.is_symlink()
            or len(shard_manifest) > max_manifest_bytes
            or hashlib.sha256(shard_manifest).hexdigest() != entry["manifest_sha256"]
        ):
            raise ValueError("Partition manifest checksum mismatch")
        arrays, metadata = load_model_dataset(path, limits=limits)
        if "scaled_sha256" in entry:
            maximum = (limits or DatasetLimits()).max_bytes
            with (path / "scaled.npz").open("rb") as stream:
                body = stream.read(maximum + 1)
            if len(body) > maximum or hashlib.sha256(body).hexdigest() != entry["scaled_sha256"]:
                raise ValueError("Scaled partition checksum/size mismatch")
            inventory = {
                "x": {
                    "shape": list(arrays["sequence_x"].shape),
                    "dtype": arrays["sequence_x"].dtype.str,
                },
                "mask": {
                    "shape": list(arrays["sequence_mask"].shape),
                    "dtype": arrays["sequence_mask"].dtype.str,
                },
            }
            scaled = _load_npz(body, inventory, limits or DatasetLimits())
            arrays["sequence_x_scaled"], arrays["sequence_scaled_mask"] = (
                scaled["x"],
                scaled["mask"],
            )
        metadata["partition"] = entry
        yield arrays, metadata
