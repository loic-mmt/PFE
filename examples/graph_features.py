"""Tiny synthetic graph: role channels, topology, dynamics and immutable GNN shards."""

from tempfile import TemporaryDirectory

import pandas as pd

from pfe import (
    build_graph_dataset,
    compute_graph_features,
    compute_graph_pair_features,
    import_graph_relations,
    iter_graph_partitions,
    iter_saved_partitions,
    save_partitioned_graph_dataset,
)


def main():
    at = "2026-01-02T10:00:00Z"
    ids = ["A", "B", "C", "D"]
    entities = [dict(alias=key, entity_id=key, node_type="company", available_at=at) for key in ids]
    relations = [
        dict(
            relationship_id=f"{owner}-{peer}",
            company_id=owner,
            related_entity_id=peer,
            relation_type="supplier",
            source_id="synthetic",
            revision_id="1",
            available_at=at,
            source_document_id="fixture:graph",
            evidence_quote="Synthetic annotated relationship.",
            qualification_reference="authored-fixture",
            revenue_dependency_pct=weight,
            weight_period="FY2025",
        )
        for owner, peer, weight in (("A", "B", 20), ("B", "C", 30), ("A", "D", 10))
    ]
    history = import_graph_relations(relations, entities)
    observations = pd.DataFrame(
        [
            dict(
                entity_id=key,
                observation_date=at,
                available_at=at,
                published_at=None,
                value=value,
                metric="momentum_20",
                unit="fraction",
                source_id="synthetic",
                revision_id="1",
                schema_version="1",
            )
            for key, value in zip(ids, [0.01, 0.02, 0.03, 0.04], strict=True)
        ]
    )
    decisions = pd.date_range(at, periods=4, freq="min")
    features = compute_graph_features(
        decisions,
        history,
        entity_ids=["A"],
        node_features=observations,
        node_feature_units={"momentum_20": "fraction"},
        groups=("channels", "two_hop", "topology", "dynamics"),
    )
    assert features.iloc[0]["graph.neighbors.customers.momentum_20.mean"] == 0.03
    pairs = compute_graph_pair_features(decisions, history, pairs=[("B", "D")])
    assert pairs.iloc[0]["graph.pair.shared_suppliers"] == 1
    registry = [
        dict(name="momentum_20", metric="momentum_20", unit="fraction", source_id="synthetic")
    ]
    dataset = build_graph_dataset(
        decisions, observations, history, registry=registry, entity_ids=["A"]
    )
    assert len(next(dataset.snapshots()).node_ids) == 4
    with TemporaryDirectory(prefix="pfe-graph-") as directory:
        parts = iter_graph_partitions(
            decisions,
            [observations],
            history,
            entity_ids=["A"],
            registry=registry,
            partition_size=2,
            lookback=2,
        )
        manifest = save_partitioned_graph_dataset(
            directory, parts, history, max_archive_bytes=1_048_576
        )
        count = sum(len(meta["graphs"]) for _, meta in iter_saved_partitions(directory))
        print(f"{len(manifest['shards'])} shards, {count} graph snapshots, no downloads")


if __name__ == "__main__":
    main()
