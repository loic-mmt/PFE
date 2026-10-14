"""Immutable, checksummed NPZ/JSON artifacts; no pickle or framework install."""

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from pfe.datasets.core import DatasetLimits, check_bytes


def _publish(path, body):
    if path.is_symlink():
        raise ValueError("Dataset artifact must not be a symlink")
    if path.exists():
        with path.open("rb") as stream:
            if stream.read(len(body) + 1) != body:
                raise ValueError("Dataset artifact is immutable; choose another output directory")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open("rb") as stream:
                if stream.read(len(body) + 1) != body:
                    raise ValueError("Concurrent dataset artifact differs") from None
    finally:
        temporary.unlink(missing_ok=True)


def _check_sequence_panel(sequences, panel, fingerprint):
    if (
        sequences.panel_fingerprint != fingerprint
        or sequences.registry_fingerprint != panel.registry.fingerprint
    ):
        raise ValueError("Sequence/panel provenance mismatch")
    lookback = sequences.lookback
    count, entities, features = panel.values.shape
    if type(lookback) is not int or not 1 <= lookback <= count:
        raise ValueError("Sequence lookback differs from panel history")
    samples = len(sequences.end_indices)
    if (
        sequences.x.shape != (samples, lookback, features)
        or sequences.mask.shape != sequences.x.shape
    ):
        raise ValueError("Sequence feature axes differ from panel")
    for name in ("entity_indices", "decision_ns", "window_start_ns"):
        if getattr(sequences, name).shape != (samples,):
            raise ValueError("Sequence sample axes differ from panel")
    for name in ("end_indices", "entity_indices"):
        if not np.issubdtype(getattr(sequences, name).dtype, np.integer):
            raise ValueError("Sequence indices must be integers")
    for sample, (end, entity) in enumerate(
        zip(sequences.end_indices, sequences.entity_indices, strict=True)
    ):
        if not lookback - 1 <= end < count or not 0 <= entity < entities:
            raise ValueError("Sequence indices differ from panel")
        start = end - lookback + 1
        if (
            sequences.decision_ns[sample] != panel.decisions_ns[end]
            or sequences.window_start_ns[sample] != panel.decisions_ns[start]
        ):
            raise ValueError("Sequence clocks differ from panel")
        expected_mask = panel.mask[start : end + 1, entity]
        expected_x = np.where(expected_mask, panel.values[start : end + 1, entity], 0.0)
        if not np.array_equal(sequences.mask[sample], expected_mask) or not np.array_equal(
            sequences.x[sample], expected_x
        ):
            raise ValueError("Sequence feature values differ from panel")


def save_model_dataset(
    directory,
    panel,
    *,
    sequences=None,
    graphs=(),
    events=None,
    split=None,
    scaler=None,
    configuration=None,
    limits=None,
):
    limits = limits or DatasetLimits()
    panel_fingerprint = panel.fingerprint
    arrays = {
        "panel_" + name: getattr(panel, name)
        for name in (
            "values",
            "mask",
            "available_at_ns",
            "observation_at_ns",
            "provenance_codes",
            "decisions_ns",
        )
    }
    metadata = dict(
        schema_version="1",
        panel_fingerprint=panel_fingerprint,
        entity_ids=panel.entity_ids,
        registry=panel.registry.to_dict(),
        provenance=panel.provenance,
        configuration=configuration or {},
        complete=True,
        graphs=[],
        semantics="selected_universe_causal_inputs; labels_separate; missing_values_masked",
    )
    if sequences is not None:
        _check_sequence_panel(sequences, panel, panel_fingerprint)
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
        ):
            arrays["sequence_" + name] = getattr(sequences, name)
        metadata["sequence"] = dict(
            lookback=sequences.lookback,
            fingerprint=sequences.fingerprint,
            layout="sample,time,feature",
        )
    for i, graph in enumerate(graphs):
        if i >= limits.max_times:
            raise ValueError("Graph snapshot count exceeds budget")
        if graph.panel_fingerprint != panel_fingerprint:
            raise ValueError("Graph/panel provenance mismatch")
        positions = np.flatnonzero(panel.decisions_ns == graph.decision_ns)
        if len(positions) != 1 or len(set(graph.node_ids)) != len(graph.node_ids):
            raise ValueError("Graph decision/nodes differ from panel")
        if not set(graph.node_ids) <= set(panel.entity_ids):
            raise ValueError("Graph node identities differ from panel")
        locations = [panel.entity_ids.index(entity) for entity in graph.node_ids]
        expected_mask = panel.mask[positions[0], locations]
        expected_x = np.where(expected_mask, panel.values[positions[0], locations], 0.0)
        if not np.array_equal(graph.mask, expected_mask) or not np.array_equal(graph.x, expected_x):
            raise ValueError("Graph feature values differ from panel")
        for name in ("x", "mask", "edge_index", "edge_weight", "edge_available_at_ns"):
            arrays[f"graph_{i}_" + name] = getattr(graph, name)
        if graph.edge_attr is not None:
            if (
                graph.edge_attr_mask is None
                or graph.edge_attr.shape != graph.edge_attr_mask.shape
                or graph.edge_attr.shape != (len(graph.edge_weight), len(graph.edge_attr_names))
            ):
                raise ValueError("Graph edge attribute axes differ")
            if graph.edge_attr_mask.dtype != np.bool_ or not np.isfinite(graph.edge_attr).all():
                raise ValueError("Graph edge attributes need finite values and boolean masks")
            arrays[f"graph_{i}_edge_attr"] = graph.edge_attr
            arrays[f"graph_{i}_edge_attr_mask"] = graph.edge_attr_mask
        metadata["graphs"].append(
            dict(
                decision_ns=graph.decision_ns,
                node_ids=graph.node_ids,
                node_types=graph.node_types,
                edge_types=graph.edge_types,
                source_documents=graph.edge_source_documents,
                edge_weight_kind=graph.edge_weight_kind,
                edge_owner_entity_ids=graph.edge_owner_entity_ids,
                edge_provenance=graph.edge_provenance,
                panel_fingerprint=graph.panel_fingerprint,
                edge_attr_names=graph.edge_attr_names,
            )
        )
        check_bytes(limits, *arrays.values())
    if events is not None:
        if events.entity_ids != panel.entity_ids or not np.array_equal(
            events.decision_ns, panel.decisions_ns
        ):
            raise ValueError("Event axes differ from panel")
        for name in (
            "decision_ns",
            "event_ids",
            "event_version_ids",
            "event_type",
            "event_status",
            "mask",
            "first_available_at_ns",
            "age_seconds",
            "magnitudes",
            "magnitude_mask",
            "magnitude_units",
            "source_documents",
            "embeddings",
            "embedding_mask",
        ):
            arrays["events_" + name] = getattr(events, name)
        metadata["events"] = dict(
            entity_ids=events.entity_ids,
            model_revision=events.model_revision,
            layout="decision,entity,event,embedding_dimension",
        )
        from pfe.extraction.schemas import EVENT_TYPES

        metadata["events"].update(
            event_types=list(EVENT_TYPES),
            event_statuses=["confirmed", "announced_plan", "rumor"],
            padding_code=-1,
        )
    if split is not None:
        if sequences is None or split.dataset_fingerprint != sequences.fingerprint:
            raise ValueError("Split belongs to another dataset")
        for name in ("train", "validation", "test", "purged"):
            arrays["split_" + name] = getattr(split, name)
        metadata["split"] = split.boundaries
    if scaler is not None:
        if sequences is None or scaler.registry_fingerprint != panel.registry.fingerprint:
            raise ValueError("Scaler feature registry mismatch")
        if split is None:
            raise ValueError("Persisted scaler requires its training split")
        from pfe.datasets.core import digest_metadata

        expected = digest_metadata(
            {"dataset": sequences.fingerprint, "train": split.train.tolist()}
        )
        if scaler.training_fingerprint != expected:
            raise ValueError("Scaler was fitted on a different training split")
        metadata["scaler"] = scaler.to_dict()
        arrays["sequence_x_scaled"], arrays["sequence_scaled_mask"] = scaler.transform(sequences)
    check_bytes(limits, *arrays.values())
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Dataset arrays must not contain objects or require pickle")
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    body = stream.getvalue()
    metadata["arrays"] = {
        name: {"shape": list(a.shape), "dtype": a.dtype.str} for name, a in arrays.items()
    }
    metadata["data_sha256"] = hashlib.sha256(body).hexdigest()
    manifest = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False).encode()
    if len(body) + len(manifest) > limits.max_bytes:
        raise ValueError("Dataset artifact plus manifest exceeds max_bytes")
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Dataset directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    _publish(directory / "arrays.npz", body)
    _publish(directory / "manifest.json", manifest)
    return metadata


def load_model_dataset(directory, *, limits=None):
    limits = limits or DatasetLimits()
    directory = Path(directory)

    def read(name):
        path = directory / name
        if path.is_symlink():
            raise ValueError("Dataset artifact must not be a symlink")
        with path.open("rb") as stream:
            body = stream.read(limits.max_bytes + 1)
        if len(body) > limits.max_bytes:
            raise ValueError("Dataset file exceeds byte budget")
        return body

    manifest, body = read("manifest.json"), read("arrays.npz")
    if len(manifest) + len(body) > limits.max_bytes:
        raise ValueError("Dataset total exceeds byte budget")
    metadata = json.loads(manifest)
    if (
        metadata.get("schema_version") != "1"
        or metadata.get("complete") is not True
        or hashlib.sha256(body).hexdigest() != metadata.get("data_sha256")
    ):
        raise ValueError("Dataset manifest/checksum invalid")
    return _load_npz(body, metadata["arrays"], limits), metadata


def _load_npz(body, inventory, limits):
    """Validate NPY allocation headers before reading any archive array."""
    if len(body) > limits.max_bytes:
        raise ValueError("Dataset file exceeds byte budget")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        if sum(item.file_size for item in archive.infolist()) > limits.max_bytes:
            raise ValueError("Expanded dataset exceeds byte budget")
        entries = archive.infolist()
        if len({item.filename for item in entries}) != len(entries) or {
            item.filename for item in entries
        } != {name + ".npy" for name in inventory}:
            raise ValueError("Dataset ZIP inventory differs from manifest")
        for item in entries:
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported bounded NPY header")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, dtype = reader(stream)
                size = math.prod(shape) * dtype.itemsize
                if (
                    dtype.hasobject
                    or size > limits.max_bytes
                    or size != item.file_size - stream.tell()
                ):
                    raise ValueError("Dataset NPY declared allocation invalid or oversized")
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if set(arrays) != set(inventory):
        raise ValueError("Dataset array inventory differs from manifest")
    for name, array in arrays.items():
        expected = inventory[name]
        if list(array.shape) != expected["shape"] or array.dtype.str != expected["dtype"]:
            raise ValueError("Dataset array shape/dtype differs from manifest")
    check_bytes(limits, *arrays.values())
    return arrays
