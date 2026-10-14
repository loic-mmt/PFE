"""Assemble typed graph inputs and stream snapshots without a DL dependency."""

import hashlib
from dataclasses import dataclass

import pandas as pd

from pfe.data.contracts import _instant, normalize_observations
from pfe.datasets.core import (
    DatasetLimits,
    FeatureRegistry,
    build_feature_panel,
    feature_frame_to_observations,
)
from pfe.datasets.partitioned import iter_panel_partitions, save_partitioned_dataset
from pfe.datasets.structured import iter_graph_tensors
from pfe.features.graph import compute_graph_features, decision_grid


def graph_entity_ids(history, decisions, entity_ids, *, include_context=True):
    index = decision_grid(decisions, entity_ids)
    if not len(index):
        raise ValueError("Graph dataset needs decisions")
    if type(include_context) is not bool:
        raise ValueError("include_context must be boolean")
    context = set()
    if include_context:
        context = set(
            history.nodes.loc[
                history.nodes.available_at.map(
                    lambda value: _instant(value, "node_available_at") <= index[-1]
                ),
                "entity_id",
            ]
        )
    return tuple(sorted(set(entity_ids) | context))


def graph_feature_observations(
    decisions, history, observations, *, entity_ids, feature_units, options=None
):
    """Selected graph features -> canonical inputs with conservative decision gates.

    Historical replay is not certified PIT. Per-decision source references hash
    only graph versions already known, so future evidence cannot change old proofs.
    """
    if not isinstance(feature_units, dict) or not feature_units:
        raise ValueError("Select explicit graph feature names and units")
    options = dict(options or {})
    if any("neighbors." in name for name in feature_units) and not options.get(
        "node_feature_units"
    ):
        raise ValueError("Declare node_feature_units for graph feature assembly")
    frame = compute_graph_features(
        decisions, history, entity_ids=entity_ids, node_features=observations, **options
    )
    if set(feature_units) - set(frame):
        raise ValueError("Selected graph feature name is unavailable")
    registry = FeatureRegistry(
        [
            dict(name=name, metric=name, unit=unit, source_id="pfe_graph")
            for name, unit in feature_units.items()
        ]
    )
    captures = []
    for decision, selected in frame.groupby(level="decision_at", sort=True):
        known = history.edges.loc[
            history.edges.available_at.map(
                lambda value, at=decision: _instant(value, "edge_available_at") <= at
            )
        ]
        proof = hashlib.sha256(
            known.sort_values(["edge_id", "available_at"])
            .to_json(orient="records", date_format="iso")
            .encode()
        ).hexdigest()
        clocks = pd.DataFrame({name: decision for name in feature_units}, index=selected.index)
        captures.append(
            feature_frame_to_observations(
                selected,
                registry,
                available_at=clocks,
                revision_id="graph-features-1:" + proof,
                source_document_id="graph-history:" + proof,
            )
        )
    if not captures:
        raise ValueError("Graph feature decisions must not be empty")
    return normalize_observations(pd.concat(captures, ignore_index=True)), registry, frame


@dataclass(frozen=True)
class GraphDataset:
    panel: object
    history: object
    graph_features: pd.DataFrame | None
    target_entity_ids: tuple[str, ...]
    limits: DatasetLimits
    observations: pd.DataFrame

    def snapshots(self, **options):
        return iter_graph_tensors(self.panel, self.history, limits=self.limits, **options)


def build_graph_dataset(
    decisions,
    observations,
    history,
    *,
    registry,
    entity_ids,
    include_context=True,
    graph_feature_units=None,
    graph_feature_options=None,
    limits=None,
):
    """Combine company/context node inputs, selected graph metrics and typed edges.

    Context nodes with unavailable features remain present with false masks.
    Targets are explicit and separate from neighbor/context nodes.
    """
    limits = limits or DatasetLimits()
    decisions = decision_grid(decisions, entity_ids)
    entities = graph_entity_ids(history, decisions, entity_ids, include_context=include_context)
    registry = registry if isinstance(registry, FeatureRegistry) else FeatureRegistry(registry)
    observations = normalize_observations(observations)
    frame = None
    if graph_feature_units:
        derived, extra, frame = graph_feature_observations(
            decisions,
            history,
            observations,
            entity_ids=entities,
            feature_units=graph_feature_units,
            options=graph_feature_options,
        )
        observations = normalize_observations(pd.concat([observations, derived], ignore_index=True))
        registry = FeatureRegistry([*registry.features, *extra.features])
    panel = build_feature_panel(
        decisions, observations, entity_ids=entities, registry=registry, limits=limits
    )
    return GraphDataset(panel, history, frame, tuple(entity_ids), limits, observations)


def iter_graph_partitions(
    decisions,
    observation_batches,
    history,
    *,
    entity_ids,
    registry,
    include_context=True,
    partition_size=128,
    lookback=1,
    limits=None,
):
    """Stream canonical node/graph inputs; keep lookback panels, nonoverlapping cores."""
    decisions = decision_grid(decisions, entity_ids)
    entities = graph_entity_ids(history, decisions, entity_ids, include_context=include_context)
    yield from iter_panel_partitions(
        decisions,
        observation_batches,
        entity_ids=entities,
        registry=registry,
        partition_size=partition_size,
        lookback=lookback,
        limits=limits,
    )


def save_partitioned_graph_dataset(
    directory, partitions, history, *, configuration=None, **options
):
    """One checked graph snapshot per core decision; shared immutable shard format."""
    return save_partitioned_dataset(
        directory, partitions, graph_history=history, configuration=configuration, **options
    )
