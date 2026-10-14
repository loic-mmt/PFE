"""Ragged GNN snapshots and bounded Event Encoder inputs on known histories."""

from dataclasses import dataclass

import numpy as np

from pfe.data.contracts import _instant
from pfe.data.events import events_asof
from pfe.data.graph import graph_asof
from pfe.datasets.core import NAT, DatasetLimits, check_bytes
from pfe.extraction.schemas import EVENT_TYPES


@dataclass(frozen=True)
class GraphTensor:
    decision_ns: int
    node_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    x: np.ndarray
    mask: np.ndarray
    edge_index: np.ndarray
    edge_types: tuple[str, ...]
    edge_weight: np.ndarray
    edge_available_at_ns: np.ndarray
    edge_source_documents: tuple[str, ...]
    edge_weight_kind: tuple[str, ...]
    edge_owner_entity_ids: tuple[str, ...]
    edge_provenance: tuple[dict, ...]
    panel_fingerprint: str
    edge_attr: np.ndarray | None = None
    edge_attr_mask: np.ndarray | None = None
    edge_attr_names: tuple[str, ...] = ()


EDGE_ATTRIBUTE_NAMES = (
    "revenue_dependency",
    "supply_dependency",
    "exposure",
    "exposure_lower",
    "exposure_upper",
    "evidence_age_seconds",
    "availability_age_seconds",
    "point_in_time",
)


def _edge_attributes(edges, decision):
    import json

    values = np.full((len(edges), len(EDGE_ATTRIBUTE_NAMES)), np.nan)
    for i, edge in enumerate(edges.itertuples()):
        values[i, :3] = [
            np.nan if v is None else float(v) / 100
            for v in (
                edge.revenue_dependency_pct,
                edge.supply_dependency_pct,
                edge.exposure_percentage,
            )
        ]
        record = json.loads(edge.record_json)
        bounds = record.get("exposure_percentage")
        if isinstance(bounds, dict):
            for j, name in enumerate(("lower_bound", "upper_bound"), 3):
                if bounds.get(name) is not None:
                    values[i, j] = float(bounds[name]) / 100
        values[i, 5] = (decision - _instant(edge.evidence_at, "evidence_at").value) / 1e9
        values[i, 6] = (decision - _instant(edge.available_at, "available_at").value) / 1e9
        values[i, 7] = float(edge.point_in_time)
    mask = np.isfinite(values)
    return np.where(mask, values, 0.0), mask


def iter_graph_tensors(panel, history, *, max_age=None, limits=None):
    """Only known graph nodes present in the explicit panel universe are emitted.

    Edges stay directed and typed. Edge weight is the known source-owner
    percentage divided by 100, or NaN; no renormalization of partial networks.
    """
    limits = limits or DatasetLimits()
    fingerprint = panel.fingerprint
    for i, decision in enumerate(panel.decisions_ns):
        import pandas as pd

        graph = graph_asof(history, pd.Timestamp(decision, tz="UTC"), max_age=max_age)
        nodes = graph.nodes.loc[graph.nodes.entity_id.isin(panel.entity_ids)].sort_values(
            "entity_id"
        )
        ids = tuple(nodes.entity_id)
        mapping = {entity: j for j, entity in enumerate(ids)}
        locations = [panel.entity_ids.index(entity) for entity in ids]
        edges = graph.edges.loc[
            graph.edges.source_entity_id.isin(ids) & graph.edges.target_entity_id.isin(ids)
        ]
        if len(edges) > limits.max_edges:
            raise ValueError("Graph edge count exceeds budget")
        index = (
            np.array(
                [
                    [mapping[r.source_entity_id], mapping[r.target_entity_id]]
                    for r in edges.itertuples()
                ],
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .T
        )
        weights, kinds = [], []
        for r in edges.itertuples():
            values = [
                getattr(r, field)
                for field in (
                    "exposure_percentage",
                    "revenue_dependency_pct",
                    "supply_dependency_pct",
                )
            ]
            finite = [float(v) for v in values if v is not None and np.isfinite(v)]
            weights.append(finite[0] / 100 if len(finite) == 1 else np.nan)
            fields = ("exposure_percentage", "revenue_dependency_pct", "supply_dependency_pct")
            kinds.append(
                next(
                    (
                        name
                        for name, v in zip(fields, values, strict=True)
                        if v is not None and np.isfinite(v)
                    ),
                    "unknown",
                )
                if len(finite) == 1
                else "ambiguous_or_unknown"
            )
        weight = np.asarray(weights, dtype=float)
        available = np.array(
            [_instant(v, "edge_available_at").value for v in edges.available_at], dtype=np.int64
        )
        x, mask = panel.values[i, locations], panel.mask[i, locations]
        x = np.where(mask, x, 0.0)
        edge_attr, attr_mask = _edge_attributes(edges, int(decision))
        check_bytes(limits, x, mask, index, weight, available, edge_attr, attr_mask)
        yield GraphTensor(
            int(decision),
            ids,
            tuple(nodes.node_type),
            x,
            mask,
            index,
            tuple(edges.relation_type),
            weight,
            available,
            tuple(edges.source_document_id),
            tuple(kinds),
            tuple(edges.owner_entity_id),
            tuple(
                {
                    name: (None if isinstance(value, float) and not np.isfinite(value) else value)
                    for name in (
                        "edge_id",
                        "fact_id",
                        "source_id",
                        "revision_id",
                        "evidence_quote",
                        "weight_scope",
                        "weight_period",
                        "review_status",
                        "point_in_time",
                    )
                    for value in [getattr(row, name)]
                }
                for row in edges.itertuples()
            ),
            fingerprint,
            edge_attr,
            attr_mask,
            EDGE_ATTRIBUTE_NAMES,
        )


@dataclass(frozen=True)
class EventTensor:
    decision_ns: np.ndarray
    entity_ids: tuple[str, ...]
    event_ids: np.ndarray
    event_version_ids: np.ndarray
    event_type: np.ndarray
    event_status: np.ndarray
    mask: np.ndarray
    first_available_at_ns: np.ndarray
    age_seconds: np.ndarray
    magnitudes: np.ndarray
    magnitude_mask: np.ndarray
    magnitude_units: np.ndarray
    source_documents: np.ndarray
    embeddings: np.ndarray
    embedding_mask: np.ndarray
    model_revision: str | None


def build_event_tensor(
    decisions,
    history,
    *,
    entity_ids,
    max_events=16,
    enrichments=(),
    model_revision=None,
    embedding_dimensions=0,
    model_available_at=None,
    limits=None,
):
    """Latest known linked events, padding masks and separately versioned vectors.

    A historical embedding requires a declared model availability AND response
    availability no later than the decision. No backdating today's inference.
    """
    from pfe.features.graph import decision_grid

    limits = limits or DatasetLimits()
    index = decision_grid(decisions, entity_ids)
    if type(max_events) is not int or not 1 <= max_events <= limits.max_events:
        raise ValueError("Event sequence length exceeds budget")
    if type(embedding_dimensions) is not int or not 0 <= embedding_dimensions <= 2048:
        raise ValueError("Embedding dimension must be in [0,2048]")
    if enrichments and (
        not model_revision or model_available_at is None or embedding_dimensions < 1
    ):
        raise ValueError(
            "Embedding inputs require model revision, model availability and dimensions"
        )
    if len(index) > limits.max_times or len(entity_ids) > limits.max_entities:
        raise ValueError("Event axes exceed dataset budgets")
    shape = (len(index), len(entity_ids), max_events)
    if np.prod(shape) * (2600 + embedding_dimensions * 8) > limits.max_bytes:
        raise ValueError("Event allocation exceeds max_bytes")
    ids = np.full(shape, "", dtype="U160")
    versions = np.full(shape, "", dtype="U160")
    docs = np.full(shape, "", dtype="U160")
    types, statuses = np.full(shape, -1, dtype=np.int16), np.full(shape, -1, dtype=np.int16)
    mask, emask = np.zeros(shape, bool), np.zeros(shape, bool)
    available = np.full(shape, NAT, dtype=np.int64)
    ages, magnitudes = np.zeros(shape), np.zeros(shape)
    magnitude_mask = np.zeros(shape, dtype=bool)
    units = np.full(shape, "", dtype="U128")
    vectors = np.zeros((*shape, embedding_dimensions))
    model_time = (
        _instant(model_available_at, "model_available_at")
        if model_available_at is not None
        else None
    )
    for i, decision in enumerate(index):
        snapshot = events_asof(history, decision)
        for j, entity in enumerate(entity_ids):
            links = snapshot.links.loc[snapshot.links.entity_id.eq(entity)]
            events = snapshot.events.loc[snapshot.events.event_id.isin(links.event_id)].copy()
            if events.empty:
                continue
            first = {
                r.event_id: max(
                    _instant(r.first_available_at, "first_available_at"),
                    _instant(
                        links.loc[links.event_id.eq(r.event_id), "first_available_at"].min(),
                        "link_available_at",
                    ),
                )
                for r in events.itertuples()
            }
            events["_first"] = events.event_id.map(first)
            events = events.sort_values(["_first", "event_id"], kind="stable").tail(max_events)
            for k, event in enumerate(events.itertuples()):
                if (
                    len(event.event_id) > 160
                    or len(event.source_document_id) > 160
                    or event.magnitude_unit is not None
                    and len(event.magnitude_unit) > 128
                ):
                    raise ValueError("Event identifier/unit exceeds fixed string capacity")
                ids[i, j, k], docs[i, j, k] = event.event_id, event.source_document_id
                if len(event.event_version_id) > 160:
                    raise ValueError("Event version identifier exceeds fixed string capacity")
                versions[i, j, k] = event.event_version_id
                types[i, j, k] = EVENT_TYPES.index(event.event_type)
                statuses[i, j, k] = ("confirmed", "announced_plan", "rumor").index(
                    event.event_status
                )
                mask[i, j, k], available[i, j, k] = True, first[event.event_id].value
                ages[i, j, k] = (decision - first[event.event_id]).total_seconds()
                magnitude_mask[i, j, k] = event.magnitude is not None and np.isfinite(
                    event.magnitude
                )
                magnitudes[i, j, k] = event.magnitude if magnitude_mask[i, j, k] else 0.0
                units[i, j, k] = event.magnitude_unit or ""
                compatible = [
                    r
                    for r in enrichments
                    if r.get("status") == "accepted"
                    and r.get("is_model_output") is True
                    and r.get("kind") == "embedding"
                    and r.get("model", {}).get("model_revision") == model_revision
                    and r.get("source_document_id") == event.source_document_id
                    and r.get("text_sha256") == event.text_sha256
                    and _instant(r["available_at"], "embedding_available_at") <= decision
                ]
                if compatible and model_time is not None and model_time <= decision:
                    output = max(
                        compatible, key=lambda r: _instant(r["available_at"], "available_at")
                    )
                    vector = np.asarray(output["value"]["embedding"], dtype=float)
                    if vector.shape != (embedding_dimensions,) or not np.isfinite(vector).all():
                        raise ValueError("Embedding dimension or finite-value mismatch")
                    vectors[i, j, k], emask[i, j, k] = vector, True
    check_bytes(
        limits,
        ids,
        versions,
        docs,
        types,
        statuses,
        mask,
        available,
        ages,
        magnitudes,
        magnitude_mask,
        units,
        vectors,
        emask,
    )
    return EventTensor(
        index.as_unit("ns").asi8,
        tuple(entity_ids),
        ids,
        versions,
        types,
        statuses,
        mask,
        available,
        ages,
        magnitudes,
        magnitude_mask,
        units,
        docs,
        vectors,
        emask,
        model_revision,
    )
