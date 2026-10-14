"""Small numerical networks: direction, timing, centralities, dynamics and GNN shards."""

import numpy as np
import pandas as pd
import pytest

from pfe.data.graph import graph_asof
from pfe.data.graph_inputs import (
    bind_graph_observations,
    entity_registry_from_security_master,
    import_graph_relations,
)
from pfe.datasets.graph import (
    build_graph_dataset,
    iter_graph_partitions,
    save_partitioned_graph_dataset,
)
from pfe.datasets.partitioned import iter_saved_partitions
from pfe.features.graph import compute_graph_features
from pfe.features.graph_analytics import GraphFeatureLimits, compute_graph_pair_features

AT = "2026-01-02T10:00:00Z"


def entities(*ids):
    return [
        dict(alias=key, entity_id=key, node_type="company", available_at="2026-01-01T00:00:00Z")
        for key in ids
    ]


def relation(owner, peer, kind="supplier", **changes):
    return (
        dict(
            relationship_id=f"{owner}-{peer}-{kind}",
            company_id=owner,
            related_entity_id=peer,
            relation_type=kind,
            source_id="fixture",
            revision_id="1",
            available_at=AT,
            source_document_id="doc:fixture",
            evidence_quote="Synthetic documented relation.",
            qualification_reference="synthetic-annotation",
            weight_period="FY2025",
        )
        | changes
    )


def observations(ids, values, period=AT):
    return pd.DataFrame(
        [
            dict(
                entity_id=key,
                observation_date=period,
                available_at=AT,
                published_at=None,
                value=float(value),
                metric="momentum_20",
                unit="fraction",
                source_id="fixture",
                revision_id="1",
                schema_version="1",
            )
            for key, value in zip(ids, values, strict=True)
        ]
    )


def test_intraday_channels_and_future_invariance():
    rows = [
        relation("A", "S", "customer", supply_dependency_pct=25),
        relation("A", "C", revenue_dependency_pct=50),
        relation("A", "R", "competitor"),
    ]
    graph = import_graph_relations(rows, entities("A", "S", "C", "R"))
    points = observations(["S", "C", "R"], [0.1, 0.2, 0.3], period="2026-01-02T11:00:00+01:00")
    options = dict(
        entity_ids=["A"],
        node_features=points,
        node_feature_units={"momentum_20": "fraction"},
        groups=("neighbors", "channels"),
    )
    result = compute_graph_features([AT], graph, **options)
    row = result.iloc[0]
    for channel, value in (("suppliers", 0.1), ("customers", 0.2), ("competitors", 0.3)):
        assert row[f"graph.neighbors.{channel}.momentum_20.mean"] == pytest.approx(value)
    assert row["graph.neighbors.suppliers.momentum_20.known_weighted_sum"] == pytest.approx(0.025)
    assert np.isnan(row["graph.neighbors.competitors.momentum_20.known_weighted_sum"])
    future = relation("A", "F", available_at="2026-02-01T00:00:00Z")
    changed = import_graph_relations(rows + [future], entities("A", "S", "C", "R", "F"))
    pd.testing.assert_frame_equal(result, compute_graph_features([AT], changed, **options))
    points.loc[0, "observation_date"] = "2026-01-02T10:01:00Z"
    assert np.isnan(
        compute_graph_features([AT], graph, **options).iloc[0][
            "graph.neighbors.suppliers.momentum_20.mean"
        ]
    )


def test_chain_centralities_pagerank_and_two_hop_coefficients():
    graph = import_graph_relations(
        [
            relation("B", "A", "customer", supply_dependency_pct=20),
            relation("C", "B", "customer", supply_dependency_pct=30),
        ],
        entities("A", "B", "C"),
    )
    result = compute_graph_features(
        [AT], graph, entity_ids=["A", "B", "C"], groups=("topology", "two_hop")
    )
    rows = result.xs(pd.Timestamp(AT))
    assert rows.loc["A", "graph.topology.harmonic"] == 0.75
    assert rows.loc["B", "graph.topology.betweenness"] == 0.5
    assert rows["graph.topology.pagerank"].sum() == pytest.approx(1)
    assert (
        rows.loc["C", "graph.topology.pagerank"]
        > rows.loc["B", "graph.topology.pagerank"]
        > rows.loc["A", "graph.topology.pagerank"]
    )
    assert rows.loc["C", "graph.two_hop.suppliers.unique_count"] == 1
    assert rows.loc["C", "graph.two_hop.suppliers.known_path_coefficient_sum"] == pytest.approx(
        0.06
    )
    with pytest.raises(ValueError, match="budget"):
        compute_graph_features(
            [AT], graph, entity_ids=["A"], limits=GraphFeatureLimits(max_nodes=2)
        )


def test_shared_neighbors_and_exposures():
    nodes = entities("A", "B", "S") + [
        dict(alias="JP", entity_id="JP", node_type="country", available_at="2026-01-01T00:00:00Z")
    ]
    graph = import_graph_relations(
        [
            relation("S", "A"),
            relation("S", "B"),
            relation("A", "JP", "country_exposure", exposure_percentage=20),
            relation("B", "JP", "country_exposure", exposure_percentage=30),
        ],
        nodes,
    )
    row = compute_graph_pair_features([AT], graph, pairs=[("A", "B")]).iloc[0]
    assert row["graph.pair.shared_suppliers"] == 1
    assert row["graph.pair.common_neighbors"] == 1
    assert row["graph.pair.jaccard"] == 1
    assert row["graph.pair.shared_country"] == 1


def test_changes_between_decisions_and_expiry_is_not_termination():
    first = relation("A", "B", revenue_dependency_pct=10)
    revised = first | dict(
        revision_id="2", available_at="2026-01-03T10:00:00Z", revenue_dependency_pct=20
    )
    end = revised | dict(revision_id="3", available_at="2026-01-04T10:00:00Z", state="terminated")
    graph = import_graph_relations([first, revised, end], entities("A", "B"))
    assert graph_asof(graph, "2026-01-03T12:00:00Z").edges.revenue_dependency_pct.tolist() == [20]
    row = compute_graph_features(
        ["2026-01-05T00:00:00Z"], graph, entity_ids=["A"], groups=("dynamics",)
    ).iloc[0]
    assert row["graph.dynamics.new_edges_w30D"] == 1
    assert row["graph.dynamics.weight_changes_w30D"] == 1
    assert row["graph.dynamics.terminated_edges_w30D"] == 1
    expired = import_graph_relations([first], entities("A", "B"))
    row = compute_graph_features(
        ["2026-01-05T00:00:00Z"], expired, entity_ids=["A"], groups=("dynamics",), max_age="1D"
    ).iloc[0]
    assert row["graph.dynamics.disappeared_edges_w30D"] == 1
    assert row["graph.dynamics.terminated_edges_w30D"] == 0


def test_identity_snapshot_binding_keeps_latency_and_rejects_ambiguity():
    from test_security_sources import IDENTITY, sec

    from pfe.data.security_sources import parse_security_snapshot

    master = parse_security_snapshot(
        sec("TICK"), source="sec", identities={"TICK": IDENTITY}, received_at=AT
    )
    registry = entity_registry_from_security_master(master, AT)
    assert (
        registry.binding("TICK", observation_date="2026-01-02")["entity_id"]
        == IDENTITY["entity_id"]
    )
    data = observations([IDENTITY["instrument_id"]], [1])
    bound = bind_graph_observations(data, master, AT)
    assert bound.entity_id.tolist() == [IDENTITY["entity_id"]]
    assert bound.instrument_id.tolist() == [IDENTITY["instrument_id"]]
    with pytest.raises(ValueError, match="binding"):
        bind_graph_observations(observations(["UNKNOWN"], [1]), master, AT)


def test_graph_dataset_context_edge_attributes_and_partition_replay(tmp_path):
    nodes = entities("A", "B") + [
        dict(alias="JP", entity_id="JP", node_type="country", available_at="2026-01-01T00:00:00Z")
    ]
    graph = import_graph_relations(
        [
            relation("A", "B", revenue_dependency_pct=20, supply_dependency_pct=5),
            relation("A", "JP", "country_exposure", exposure_percentage=30),
        ],
        nodes,
    )
    points = observations(["A", "B", "JP"], [0.1, 0.2, 0.3])
    registry = [
        dict(name="momentum_20", metric="momentum_20", unit="fraction", source_id="fixture")
    ]
    decisions = pd.date_range(AT, periods=4, freq="min")
    full = build_graph_dataset(
        decisions,
        points,
        graph,
        registry=registry,
        entity_ids=["A"],
        graph_feature_units={"graph.degree.supply_flow.out": "count"},
        graph_feature_options=dict(
            groups=("degree",), node_feature_units={"momentum_20": "fraction"}
        ),
    )
    assert set(full.panel.entity_ids) == {"A", "B", "JP"}
    snapshots = list(full.snapshots())
    edge = snapshots[0]
    assert edge.edge_attr.shape == (2, 8) and edge.edge_attr_mask.dtype == bool
    location = edge.edge_types.index("supplier")
    assert edge.edge_attr[location, 0] == 0.2 and edge.edge_attr[location, 1] == 0.05
    assert np.isnan(edge.edge_weight[location])

    def parts():
        return iter_graph_partitions(
            decisions,
            [points],
            graph,
            entity_ids=["A"],
            registry=registry,
            partition_size=2,
            lookback=2,
        )

    first = save_partitioned_graph_dataset(tmp_path, parts(), graph)
    assert first == save_partitioned_graph_dataset(tmp_path, parts(), graph)
    loaded = list(iter_saved_partitions(tmp_path))
    assert sum(len(meta["graphs"]) for _, meta in loaded) == 4
    assert all("graph_0_edge_attr_mask" in arrays for arrays, _ in loaded)


@pytest.mark.parametrize("partition_size", [None, 2])
def test_pipeline_graph_build_export_and_verified_replay(tmp_path, partition_size):
    from pfe import run_pipeline

    relations = tmp_path / "relations.parquet"
    points = tmp_path / "points.parquet"
    pd.DataFrame([relation("A", "B", revenue_dependency_pct=20)]).to_parquet(relations)
    observations(["A", "B"], [0.1, 0.2]).to_parquet(points)
    config = dict(
        decisions=[AT, "2026-01-02T10:01:00Z", "2026-01-02T10:02:00Z"],
        entity_ids=["A"],
        features=[
            dict(name="momentum_20", metric="momentum_20", unit="fraction", source_id="fixture")
        ],
        graph_feature_units={"graph.topology.pagerank": "score"},
        graph_feature_options=dict(groups=["topology"]),
        partition_size=partition_size,
    )
    steps = [
        dict(
            name="local",
            operation="local_tables",
            parameters=dict(paths={"relations": str(relations), "observations": str(points)}),
        ),
        dict(
            name="graph",
            operation="graph",
            needs=["local"],
            parameters=dict(relations="local.relations", entities=entities("A", "B")),
            outputs={"nodes": "graph/nodes.parquet", "edges": "graph/edges_history.parquet"},
        ),
        dict(
            name="dataset",
            operation="graph_dataset",
            needs=["local", "graph"],
            parameters=dict(
                history="graph",
                observations=["local.observations"],
                directory="datasets/gnn",
                config=config,
            ),
        ),
    ]
    root = tmp_path / "run"
    first = run_pipeline(steps, root)
    second = run_pipeline(steps, root)
    assert first["manifest"] == second["manifest"] and first["requests"] == 0
    assert (root / "datasets/gnn/manifest.json").exists()


def test_expired_reopened_relation_does_not_recount_old_termination():
    first = relation("A", "B")
    end = first | dict(available_at="2026-01-03T10:00:00Z", revision_id="2", state="terminated")
    again = first | dict(available_at="2026-01-04T10:00:00Z", revision_id="3")
    graph = import_graph_relations([first, end, again], entities("A", "B"))
    row = compute_graph_features(
        ["2026-01-07T00:00:00Z"], graph, entity_ids=["A"], groups=("dynamics",), max_age="2D"
    ).iloc[0]
    assert row["graph.dynamics.new_edges_w30D"] == 2
    assert row["graph.dynamics.disappeared_edges_w30D"] == 2
    assert row["graph.dynamics.terminated_edges_w30D"] == 1


def test_reviewed_public_excerpt_is_partial_and_not_backdated():
    import json
    from pathlib import Path

    record = json.loads((Path(__file__).parent / "fixtures/graph/public_relation.json").read_text())
    history = import_graph_relations([record], entities("company:tsmc", "company:nvidia"))
    assert graph_asof(history, "2026-10-14T16:08:21Z").edges.empty
    snapshot = graph_asof(history, record["available_at"])
    assert snapshot.edges.source_entity_id.tolist() == ["company:tsmc"]
    assert snapshot.edges.target_entity_id.tolist() == ["company:nvidia"]
    assert snapshot.edges.relation_type.tolist() == ["manufacturer"]
    assert snapshot.edges.revenue_dependency_pct.isna().all()
    assert not snapshot.edges.point_in_time.any()


def test_strict_period_policy_optional_asof_and_economic_staleness():
    history = import_graph_relations(
        [relation("A", "B"), relation("A", "C")], entities("A", "B", "C")
    )
    points = observations(["B", "C"], [0.1, 0.2])
    points.loc[1, "observation_date"] = "2026-01-01T10:00:00Z"
    options = dict(
        entity_ids=["A"],
        node_features=points,
        node_feature_units={"momentum_20": "fraction"},
        groups=("neighbors",),
    )
    strict = compute_graph_features([AT], history, **options).iloc[0]
    assert np.isnan(strict["graph.neighbors.momentum_20.mean"])
    permissive = compute_graph_features(
        [AT], history, **options, require_same_observation=False
    ).iloc[0]
    assert permissive["graph.neighbors.momentum_20.mean"] == pytest.approx(0.15)
    assert permissive["graph.quality.neighbors.momentum_20.mixed_periods"] == 1
    fresh = compute_graph_features([AT], history, **options, node_max_observation_age="1h").iloc[0]
    assert fresh["graph.neighbors.momentum_20.mean"] == 0.1


def test_structured_revision_immutability_and_bad_node_types():
    row = relation("A", "B", revenue_dependency_pct=10)
    repeated = row | dict(available_at="2026-01-03T00:00:00Z")
    graph = import_graph_relations([repeated, row], entities("A", "B"))
    assert len(graph.edges) == 1
    with pytest.raises(ValueError, match="immutable"):
        import_graph_relations(
            [row, repeated | dict(revenue_dependency_pct=20)], entities("A", "B")
        )
    with pytest.raises(ValueError, match="node type"):
        import_graph_relations([relation("A", "B", "country_exposure")], entities("A", "B"))


def test_binding_feature_and_pair_pipeline_steps(tmp_path):
    from test_security_sources import IDENTITY, sec

    from pfe import run_pipeline
    from pfe.data.security_sources import parse_security_snapshot

    master = parse_security_snapshot(
        sec("A"), source="sec", identities={"A": IDENTITY}, received_at=AT
    )
    data = dict(
        master=master,
        own=observations([IDENTITY["instrument_id"]], [0.1]),
        peers=observations(["B"], [0.2]),
        relations=pd.DataFrame([relation(IDENTITY["entity_id"], "B")]),
    )
    paths = {}
    for name, frame in data.items():
        path = tmp_path / (name + ".parquet")
        frame.to_parquet(path)
        paths[name] = str(path)
    steps = [
        dict(name="local", operation="local_tables", parameters=dict(paths=paths)),
        dict(
            name="bindings",
            operation="graph_bindings",
            needs=["local"],
            parameters=dict(
                security_master="local.master", observations="local.own", decision_at=AT
            ),
        ),
        dict(
            name="graph",
            operation="graph",
            needs=["local"],
            parameters=dict(
                security_master="local.master",
                decision_at=AT,
                aliases=entities("B"),
                relations="local.relations",
            ),
        ),
        dict(
            name="features",
            operation="graph_features",
            needs=["graph", "local"],
            parameters=dict(
                history="graph",
                observations="local.peers",
                decisions=[AT],
                entity_ids=[IDENTITY["entity_id"]],
                feature_units={"graph.neighbors.customers.momentum_20.mean": "fraction"},
                options=dict(groups=["channels"], node_feature_units={"momentum_20": "fraction"}),
            ),
        ),
        dict(
            name="pairs",
            operation="graph_pairs",
            needs=["graph"],
            parameters=dict(history="graph", decisions=[AT], pairs=[[IDENTITY["entity_id"], "B"]]),
        ),
    ]
    result = run_pipeline(steps, tmp_path / "run")
    assert result["results"]["bindings"].tables["observations"].entity_id.tolist() == [
        IDENTITY["entity_id"]
    ]
    assert result["results"]["features"].tables["observations"].value.tolist() == [0.2]
    assert result["results"]["pairs"].tables["pairs"].iloc[0]["graph.pair.mask"]
    assert result["manifest"] == run_pipeline(steps, tmp_path / "run")["manifest"]
