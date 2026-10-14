from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pfe import (
    DocumentStore,
    Extractor,
    adjudicate_fact,
    build_event_history,
    build_graph_history,
    compute_event_features,
    compute_graph_features,
    compute_statement_features,
    events_asof,
    extraction_schema,
    facts_asof,
    graph_asof,
    statements_frame,
)
from pfe.data.graph import EntityRegistry
from pfe.extraction.schemas import ENVELOPES

ENTITIES = [
    {
        "alias": alias,
        "entity_id": alias,
        "node_type": "company",
        "available_at": "2026-01-01T00:00:00Z",
    }
    for alias in ("Acme", "Beta", "Gamma")
]


def make_fact(
    tmp_path,
    task="A",
    *,
    text="Acme supplies Beta; Beta represents 20% of Acme revenue.",
    record=None,
    received="2026-01-02T10:00:00Z",
    processed="2026-01-02T10:01:00Z",
    reviewed="2026-01-02T10:02:00Z",
    approved=True,
    effective_from=None,
    effective_to=None,
):
    doc = DocumentStore(tmp_path / "docs").import_bytes(
        text.encode(), logical_id="report", received_at=received
    )
    props = extraction_schema(task)["properties"][ENVELOPES[task]]["items"]["properties"]
    values = dict.fromkeys(props)
    if task == "A":
        values.update(
            source_company="Acme",
            related_entity="Beta",
            relation_type="supplier",
            certainty="confirmed",
            effective_status="existing",
            revenue_dependency_pct=20,
        )
    elif task == "D":
        values.update(
            event_type="product_launch", event_status="confirmed", entities_involved=["Acme"]
        )
    values.update(evidence_quote=text, source_document_id=doc.source_document_id)
    values.update(record or {})
    result = Extractor(tmp_path / "extract").import_response(
        doc, task, {ENVELOPES[task]: [values]}, received_at=processed
    )
    assert not result.rejected, result.rejected
    fact = result.accepted[0]
    if reviewed is not None:
        fact = adjudicate_fact(
            fact,
            approved=approved,
            reviewer="fixture-annotator",
            reviewed_at=reviewed,
            notes="Annotated fields checked against synthetic source.",
            document=doc,
            effective_from=effective_from,
            effective_to=effective_to,
        )
    return fact, doc


def test_supply_flow_both_roles_and_review_knowledge(tmp_path):
    supplier, _ = make_fact(tmp_path)
    customer, _ = make_fact(
        tmp_path,
        record={
            "source_company": "Beta",
            "related_entity": "Acme",
            "relation_type": "customer",
            "revenue_dependency_pct": None,
            "supply_dependency_pct": None,
        },
    )
    graph = build_graph_history([supplier, customer], ENTITIES)
    before = graph_asof(graph, "2026-01-02T10:01:59Z")
    assert before.edges.empty
    after = graph_asof(graph, "2026-01-02T10:02:00Z")
    assert set(after.edges.source_entity_id) == {"Acme"}
    assert set(after.edges.target_entity_id) == {"Beta"}
    assert set(after.edges.source_role) == {"supplier", "customer"}
    assert len(facts_asof([supplier], "2026-01-02T10:02:00Z")) == 1


def test_partial_weights_neighbor_units_dates_and_future_invariance(tmp_path):
    first, _ = make_fact(tmp_path)
    second, _ = make_fact(
        tmp_path,
        text="Acme supplies Gamma.",
        record={"related_entity": "Gamma", "revenue_dependency_pct": None},
    )
    future, _ = make_fact(
        tmp_path,
        text="Acme supplies Gamma; Gamma represents 30% of Acme revenue.",
        record={"related_entity": "Gamma", "revenue_dependency_pct": 30},
        received="2026-02-01T10:00:00Z",
        processed="2026-02-01T10:01:00Z",
        reviewed="2026-02-01T10:02:00Z",
    )
    features = pd.DataFrame(
        [
            {
                "entity_id": entity,
                "observation_date": "2026-01-02",
                "published_at": "2026-01-02T10:00:00Z",
                "available_at": "2026-01-02T10:00:00Z",
                "value": value,
                "source_id": "fixture",
                "revision_id": "1",
                "metric": "momentum_20",
                "schema_version": "1",
                "unit": "fraction",
                "point_in_time": True,
                "availability_basis": "event_time",
            }
            for entity, value in (("Beta", 0.1), ("Gamma", 0.2))
        ]
    )
    kwargs = dict(
        entity_ids=["Acme"], node_features=features, node_feature_units={"momentum_20": "fraction"}
    )
    decisions = ["2026-01-02T10:03:00Z", "2026-01-03T10:03:00Z"]
    base = compute_graph_features(
        decisions, build_graph_history([first, second], ENTITIES), **kwargs
    )
    appended = compute_graph_features(
        decisions, build_graph_history([first, second, future], ENTITIES), **kwargs
    )
    pd.testing.assert_frame_equal(base, appended)
    row = base.iloc[0]
    assert row["graph.neighbors.momentum_20.mean"] == pytest.approx(0.15)
    assert row["graph.neighbors.momentum_20.known_weighted_sum"] == pytest.approx(0.02)
    assert row["graph.neighbors.momentum_20.weighted_coverage_fraction"] == 0.5
    assert row["graph.dependency.revenue.known_weight_sum"] == 0.2
    assert row["graph.dependency.revenue.concentration_known"] == pytest.approx(0.04)
    mixed = features.copy()
    mixed.loc[1, "observation_date"] = "2026-01-01"
    row = compute_graph_features(
        decisions[:1],
        build_graph_history([first, second], ENTITIES),
        **{**kwargs, "node_features": mixed},
    ).iloc[0]
    assert np.isnan(row["graph.neighbors.momentum_20.mean"])
    assert row["graph.quality.neighbors.momentum_20.mixed_periods"] == 1
    wrong_unit = features.copy()
    wrong_unit.loc[1, "unit"] = "pct"
    row = compute_graph_features(
        decisions[:1],
        build_graph_history([first, second], ENTITIES),
        **{**kwargs, "node_features": wrong_unit},
    ).iloc[0]
    assert row["graph.neighbors.momentum_20.coverage_fraction"] == 0.5


def test_country_material_ranges_null_weights_and_nodes(tmp_path):
    fact, _ = make_fact(
        tmp_path,
        "B",
        text="Acme: Taiwan copper raw material exposure 30-40% in FY2025.",
        record={
            "company": "Acme",
            "country_or_region": "Taiwan",
            "product_or_material": "copper",
            "exposure_type": "raw_material",
            "exposure_percentage": {"lower_bound": 30, "upper_bound": 40},
            "fiscal_period": "FY2025",
        },
    )
    graph = build_graph_history(
        [fact],
        ENTITIES
        + [
            {
                "alias": "Semiconductors",
                "entity_id": "industry:semis",
                "node_type": "industry",
                "available_at": "2026-01-01T00:00:00Z",
            },
            {
                "alias": "Oil",
                "entity_id": "factor:oil",
                "node_type": "factor",
                "available_at": "2026-01-01T00:00:00Z",
            },
        ],
    )
    snapshot = graph_asof(graph, "2026-01-03T00:00:00Z")
    assert set(snapshot.nodes.node_type) == {
        "company",
        "country",
        "industry",
        "commodity",
        "factor",
    }
    assert snapshot.edges.exposure_percentage.isna().all()
    assert snapshot.edges.record_json.str.contains('"upper_bound":40').all()
    row = compute_graph_features(["2026-01-03T00:00:00Z"], graph, entity_ids=["Acme"]).iloc[0]
    assert row["graph.exposure.country.count"] == 1
    assert row["graph.exposure.commodity.count"] == 1
    assert row["graph.exposure.country.known_weight_count"] == 0


def test_silence_plans_and_rejected_evidence_do_not_close_other_facts(tmp_path):
    first, _ = make_fact(tmp_path)
    plan, _ = make_fact(
        tmp_path,
        text="Acme plans to supply Beta.",
        record={
            "revenue_dependency_pct": None,
            "certainty": "possible",
            "effective_status": "planned",
        },
        received="2026-01-04T10:00:00Z",
        processed="2026-01-04T10:01:00Z",
        reviewed="2026-01-04T10:02:00Z",
    )
    graph = build_graph_history([first, plan], ENTITIES)
    assert len(graph_asof(graph, "2026-01-05T00:00:00Z").edges) == 1
    assert graph_asof(graph, "2027-01-01T00:00:00Z").edges.revenue_dependency_pct.iloc[0] == 20
    assert graph_asof(graph, "2027-01-01T00:00:00Z", max_age="30D").edges.empty
    future_unreviewed, _ = make_fact(
        tmp_path,
        text="Acme supplies Beta.",
        record={"revenue_dependency_pct": None},
        received="2026-01-04T10:00:00Z",
        processed="2026-01-04T10:01:00Z",
        reviewed=None,
    )
    assert (
        len(
            graph_asof(
                build_graph_history([first, future_unreviewed], ENTITIES), "2026-01-05T00:00:00Z"
            ).edges
        )
        == 1
    )


def test_explicit_validity_and_termination(tmp_path):
    first, _ = make_fact(tmp_path)
    ended, _ = make_fact(
        tmp_path,
        text="Acme terminated its supply relationship with Beta on 2026-01-05.",
        record={"revenue_dependency_pct": None, "effective_status": "terminated"},
        received="2026-01-04T10:00:00Z",
        processed="2026-01-04T10:01:00Z",
        reviewed="2026-01-04T10:02:00Z",
        effective_from="2026-01-05",
    )
    graph = build_graph_history([first, ended], ENTITIES)
    assert len(graph_asof(graph, "2026-01-04T11:00:00Z").edges) == 1
    assert graph_asof(graph, "2026-01-05T00:00:00Z").edges.empty


def test_rejection_is_a_review_tombstone(tmp_path):
    approved, doc = make_fact(tmp_path)
    rejected = adjudicate_fact(
        approved,
        approved=False,
        reviewer="second-annotator",
        reviewed_at="2026-01-04T10:00:00Z",
        notes="Withdraw prior interpretation.",
        document=doc,
    )
    graph = build_graph_history([approved, rejected], ENTITIES)
    assert len(graph_asof(graph, "2026-01-03T00:00:00Z").edges) == 1
    assert graph_asof(graph, "2026-01-05T00:00:00Z").edges.empty
    assert facts_asof([approved, rejected], "2026-01-05T00:00:00Z") == []


def test_late_identity_delays_join_and_ambiguous_alias_is_unresolved(tmp_path):
    fact, _ = make_fact(tmp_path)
    entities = deepcopy(ENTITIES)
    entities[1]["available_at"] = "2026-01-04T10:00:00Z"
    graph = build_graph_history([fact], entities)
    assert graph_asof(graph, "2026-01-03T00:00:00Z").edges.empty
    assert len(graph_asof(graph, "2026-01-05T00:00:00Z").edges) == 1
    ambiguous = ENTITIES + [{**ENTITIES[1], "entity_id": "another-Beta"}]
    graph = build_graph_history([fact], ambiguous)
    assert graph.edges.empty and "unresolved" in graph.issues.reason.iloc[0]
    with pytest.raises(ValueError):
        EntityRegistry([{**ENTITIES[0], "node_type": "fiction"}])


def test_event_duplicate_review_gate_and_future_invariance(tmp_path):
    first, _ = make_fact(tmp_path, "D", text="Acme launched chip 1.")
    future, _ = make_fact(
        tmp_path,
        "D",
        text="Acme launched chip 2.",
        received="2026-02-01T10:00:00Z",
        processed="2026-02-01T10:01:00Z",
        reviewed="2026-02-01T10:02:00Z",
    )
    base = build_event_history([first, first], ENTITIES)
    assert len(events_asof(base, "2026-01-03T00:00:00Z").events) == 1
    assert events_asof(base, "2026-01-02T10:01:00Z").events.empty
    decisions = ["2026-01-03T00:00:00Z"]
    a = compute_event_features(decisions, base, entity_ids=["Acme"])
    b = compute_event_features(
        decisions, build_event_history([first, future], ENTITIES), entity_ids=["Acme"]
    )
    pd.testing.assert_frame_equal(a, b)
    assert a.iloc[0]["events.count.product_launch.confirmed.7D"] == 1
    assert a.iloc[0]["events.quality.publication_coverage"] == 0


def test_explicit_cross_document_dedup_preserves_first_clock_and_rumor_status(tmp_path):
    rumor, _ = make_fact(
        tmp_path, "D", text="Acme may launch chip 1.", record={"event_status": "rumor"}
    )
    confirmed, _ = make_fact(
        tmp_path,
        "D",
        text="Acme launched chip 1.",
        received="2026-01-10T10:00:00Z",
        processed="2026-01-10T10:01:00Z",
        reviewed="2026-01-10T10:02:00Z",
    )
    ids = {rumor["fact_id"]: "launch-1", confirmed["fact_id"]: "launch-1"}
    history = build_event_history([rumor, confirmed], ENTITIES, event_ids=ids)
    frame = compute_event_features(
        ["2026-01-03T00:00:00Z", "2026-01-11T00:00:00Z"], history, entity_ids=["Acme"]
    )
    assert frame.iloc[0]["events.count.product_launch.rumor.7D"] == 1
    assert frame.iloc[1]["events.count.product_launch.confirmed.7D"] == 0
    assert frame.iloc[1]["events.count.product_launch.confirmed.30D"] == 1
    assert frame.iloc[1]["events.age_seconds"] > 8 * 86400
    assert len(history.events) == 2


def test_event_rereview_preserves_first_entity_availability(tmp_path):
    first, document = make_fact(tmp_path, "D", text="Acme launched a product.")
    reviewed = adjudicate_fact(
        first,
        approved=True,
        reviewer="second-reviewer",
        reviewed_at="2026-01-20T00:00:00Z",
        notes="Same event and attribution checked again.",
        document=document,
    )
    original = build_event_history([first], ENTITIES)
    revised = build_event_history([first, reviewed], ENTITIES)
    before = ["2026-01-03T00:00:00Z"]
    pd.testing.assert_frame_equal(
        compute_event_features(before, original, entity_ids=["Acme"]),
        compute_event_features(before, revised, entity_ids=["Acme"]),
    )
    decision = "2026-01-21T00:00:00Z"
    snapshot = events_asof(revised, decision)
    assert snapshot.events.first_available_at.iloc[0] == first["available_at"]
    assert snapshot.links.first_available_at.iloc[0] == first["available_at"]
    features = compute_event_features([decision], revised, entity_ids=["Acme"])
    assert features.iloc[0]["events.count.product_launch.confirmed.7D"] == 0
    assert (
        features.iloc[0]["events.age_seconds"]
        == (pd.Timestamp(decision) - pd.Timestamp(first["available_at"])).total_seconds()
    )


def test_magnitude_units_optional_embeddings_and_model_availability(tmp_path):
    fact, doc = make_fact(
        tmp_path,
        "D",
        text="Acme launched 2 chips.",
        record={"event_magnitude": 2, "magnitude_unit": "chips"},
    )
    history = build_event_history([fact], ENTITIES)
    output = {
        "enrichment_id": "embedding-1",
        "source_document_id": doc.source_document_id,
        "text_sha256": doc.text_sha256,
        "kind": "embedding",
        "value": {"embedding": [0.1, 0.3]},
        "model": {"model_revision": "1" * 64},
        "available_at": "2026-01-04T00:00:00Z",
        "status": "accepted",
        "is_model_output": True,
    }
    frame = compute_event_features(
        ["2026-01-03T00:00:00Z", "2026-01-05T00:00:00Z"],
        history,
        entity_ids=["Acme"],
        magnitude_units={"product_launch": "chips"},
        enrichments=[output],
        model_revision="1" * 64,
        embedding_dimensions=2,
    )
    assert np.isnan(frame.iloc[0]["events.embedding.7D.0"])
    assert frame.iloc[1]["events.embedding.7D.0"] == 0.1
    assert frame.iloc[0]["events.magnitude.product_launch.confirmed.7D.mean"] == 2
    with pytest.raises(ValueError, match="dimension"):
        compute_event_features(
            ["2026-01-05T00:00:00Z"],
            history,
            entity_ids=["Acme"],
            enrichments=[output],
            model_revision="1" * 64,
            embedding_dimensions=3,
        )
    with pytest.raises(ValueError, match="model_revision"):
        compute_event_features(
            ["2026-01-05T00:00:00Z"], history, entity_ids=["Acme"], enrichments=[output]
        )


def test_statements_keep_classes_currencies_bounds_and_review_timestamps(tmp_path):
    guide, _ = make_fact(
        tmp_path,
        "C",
        text="Acme guidance revenue USD 10-12 billion in FY2026.",
        record={
            "company": "Acme",
            "metric": "revenue",
            "statement_type": "management_guidance",
            "lower_bound": 10,
            "upper_bound": 12,
            "currency": "USD",
            "unit_scale": "billion",
            "fiscal_period_label": "FY2026",
        },
    )
    actual, _ = make_fact(
        tmp_path,
        "C",
        text="Acme revenue USD 8 billion for 2025-01-01 to 2025-12-31.",
        record={
            "company": "Acme",
            "metric": "revenue",
            "statement_type": "reported_actual",
            "value": 8,
            "currency": "USD",
            "unit_scale": "billion",
            "fiscal_period_start": "2025-01-01",
            "fiscal_period_end": "2025-12-31",
        },
    )
    data = statements_frame([guide, actual], ENTITIES)
    spec = {
        "guidance": {
            "metric": "revenue",
            "statement_type": "management_guidance",
            "currency": "USD",
            "unit_scale": "billion",
            "segment_name": None,
        }
    }
    frame = compute_statement_features(
        ["2026-01-02T10:01:00Z", "2026-01-03T00:00:00Z"],
        data,
        entity_ids=["Acme"],
        specifications=spec,
    )
    assert frame.iloc[0]["statements.guidance.mask"] == 0
    assert frame.iloc[1]["statements.guidance.lower_bound"] == 10
    assert frame.iloc[1]["statements.guidance.upper_bound"] == 12
    assert np.isnan(frame.iloc[1]["statements.guidance.value"])
    spec["guidance"]["currency"] = "EUR"
    assert (
        compute_statement_features(
            ["2026-01-03T00:00:00Z"], data, entity_ids=["Acme"], specifications=spec
        ).iloc[0]["statements.guidance.mask"]
        == 0
    )


def test_invalid_decisions_availability_and_future_snapshot(tmp_path):
    fact, _ = make_fact(tmp_path)
    graph = build_graph_history([fact], ENTITIES)
    assert graph_asof(graph, "2026-01-03T00:00:00Z", observation_date="2025-01-01").edges.empty
    with pytest.raises(ValueError, match="future"):
        graph_asof(graph, "2026-01-03T00:00:00Z", observation_date="2026-01-04")
    with pytest.raises(ValueError):
        compute_graph_features(["2026-01-03"], graph, entity_ids=["Acme"])
    bad = {**fact, "available_at": "2026-01-01T00:00:00Z"}
    with pytest.raises(ValueError, match="precedes"):
        build_graph_history([bad], ENTITIES)


def test_distinct_events_same_article_not_collapsed_when_details_null(tmp_path):
    first, doc = make_fact(
        tmp_path,
        "D",
        text="Acme launched chip 1. Acme launched chip 2.",
        record={"evidence_quote": "Acme launched chip 1."},
    )
    second, _ = make_fact(
        tmp_path, "D", text=doc.text, record={"evidence_quote": "Acme launched chip 2."}
    )
    snapshot = events_asof(
        build_event_history([first, second, first], ENTITIES), "2026-01-03T00:00:00Z"
    )
    assert len(snapshot.events) == 2
    assert len(snapshot.links) == 2


def test_structural_industry_factor_declarations_and_seven_shared_fields(tmp_path):
    from pfe import merge_graph_declarations

    registry = ENTITIES + [
        {
            "alias": label,
            "entity_id": entity,
            "node_type": kind,
            "available_at": "2026-01-01T00:00:00Z",
        }
        for label, entity, kind in (
            ("Semis", "industry:semis", "industry"),
            ("Oil", "factor:oil", "factor"),
        )
    ]
    graph = build_graph_history([], registry)
    declared = pd.DataFrame(
        [
            {
                "entity_id": "Acme",
                "observation_date": "2026-01-01",
                "published_at": None,
                "available_at": "2026-01-02T10:00:00Z",
                "ingested_at": "2026-01-02T10:00:00Z",
                "source_id": "reference-fixture",
                "revision_id": "1",
                "source_document_id": "reference-row",
                "metric": f"graph.{relation}.{target}",
                "unit": "relationship",
                "schema_version": "1",
                "point_in_time": True,
                "availability_basis": "receipt",
                "evidence_quote": "Synthetic reference row.",
                "value": {
                    "target_entity_id": target,
                    "relation_type": relation,
                    "active": True,
                    "weight_pct": weight,
                },
            }
            for target, relation, weight in (
                ("industry:semis", "industry_member", None),
                ("factor:oil", "factor_exposure", 10),
            )
        ]
    )
    merged = merge_graph_declarations(graph, declared)
    assert graph.edges.empty
    assert graph_asof(merged, "2026-01-02T09:00:00Z").edges.empty
    row = compute_graph_features(["2026-01-03T00:00:00Z"], merged, entity_ids=["Acme"]).iloc[0]
    assert row["graph.degree.industry_member.out"] == 1
    assert row["graph.exposure.factor.count"] == 1
    factor = pd.DataFrame(
        [
            {
                "entity_id": "factor:oil",
                "observation_date": "2026-01-02",
                "published_at": "2026-01-02T12:00:00Z",
                "available_at": "2026-01-02T12:00:00Z",
                "value": 0.05,
                "source_id": "factor-fixture",
                "revision_id": "1",
                "metric": "momentum_20",
                "unit": "fraction",
                "schema_version": "1",
                "point_in_time": True,
                "availability_basis": "event_time",
            }
        ]
    )
    propagated = compute_graph_features(
        ["2026-01-03T00:00:00Z"],
        merged,
        entity_ids=["Acme"],
        node_features=factor,
        node_feature_units={"momentum_20": "fraction"},
        neighbor_relations=("factor_exposure",),
        neighbor_node_types=("factor",),
    )
    assert propagated.iloc[0]["graph.neighbors.momentum_20.known_weighted_sum"] == pytest.approx(
        0.005
    )
    shared = {
        "entity_id",
        "observation_date",
        "published_at",
        "available_at",
        "value",
        "source_id",
        "revision_id",
    }
    assert shared.issubset(merged.nodes) and shared.issubset(merged.edges)
    event, _ = make_fact(tmp_path, "D", text="Acme launched a chip.")
    events = build_event_history([event], ENTITIES)
    assert shared.issubset(events.events) and shared.issubset(events.links)


def test_future_unresolved_entities_do_not_leak_through_snapshot_issues(tmp_path):
    first, _ = make_fact(tmp_path)
    future, _ = make_fact(
        tmp_path,
        text="Acme supplies Missing.",
        record={"related_entity": "Missing", "revenue_dependency_pct": None},
        received="2026-02-01T10:00:00Z",
        processed="2026-02-01T10:01:00Z",
        reviewed="2026-02-01T10:02:00Z",
    )
    history = build_graph_history([first, future], ENTITIES)
    assert history.issues.shape[0] == 1
    assert graph_asof(history, "2026-01-03T00:00:00Z").issues.empty


def test_late_restatement_of_old_period_does_not_replace_newest_actual(tmp_path):
    first, _ = make_fact(
        tmp_path,
        "C",
        text="Acme reported revenue USD 8 billion for 2025-01-01 to 2025-12-31.",
        record={
            "company": "Acme",
            "metric": "revenue",
            "statement_type": "reported_actual",
            "value": 8,
            "currency": "USD",
            "unit_scale": "billion",
            "fiscal_period_start": "2025-01-01",
            "fiscal_period_end": "2025-12-31",
        },
    )
    revised, _ = make_fact(
        tmp_path,
        "C",
        text="Acme reported revenue USD 5 billion for 2024-01-01 to 2024-12-31.",
        record={
            "company": "Acme",
            "metric": "revenue",
            "statement_type": "reported_actual",
            "value": 5,
            "currency": "USD",
            "unit_scale": "billion",
            "fiscal_period_start": "2024-01-01",
            "fiscal_period_end": "2024-12-31",
        },
        received="2026-01-05T10:00:00Z",
        processed="2026-01-05T10:01:00Z",
        reviewed="2026-01-05T10:02:00Z",
    )
    specs = {
        "actual": {
            "metric": "revenue",
            "statement_type": "reported_actual",
            "currency": "USD",
            "unit_scale": "billion",
            "segment_name": None,
        }
    }
    values = compute_statement_features(
        ["2026-01-06T00:00:00Z"],
        statements_frame([first, revised], ENTITIES),
        entity_ids=["Acme"],
        specifications=specs,
    )
    assert values.iloc[0]["statements.actual.value"] == 8


def test_simultaneous_conflicting_weights_are_missing_not_last_row_wins(tmp_path):
    first, _ = make_fact(tmp_path)
    second, _ = make_fact(
        tmp_path,
        text="Acme supplies Beta; Beta represents 30% of Acme revenue.",
        record={"revenue_dependency_pct": 30},
    )
    snapshot = graph_asof(build_graph_history([first, second], ENTITIES), "2026-01-03T00:00:00Z")
    assert len(snapshot.edges) == 1
    assert snapshot.edges.ambiguous_weights.iloc[0]
    assert pd.isna(snapshot.edges.revenue_dependency_pct.iloc[0])
    assert len(snapshot.edges.supporting_fact_ids.iloc[0]) == 2
    values = compute_graph_features(
        ["2026-01-03T00:00:00Z"],
        build_graph_history([second, first], ENTITIES),
        entity_ids=["Acme"],
    )
    assert np.isnan(values.iloc[0]["graph.dependency.revenue.known_weight_sum"])


def test_weight_periods_and_products_are_not_aggregated_together(tmp_path):
    first, _ = make_fact(
        tmp_path,
        text="Acme supplies Beta; Beta represents 20% of Acme revenue in FY2024.",
        record={"effective_period": "FY2024"},
    )
    second, _ = make_fact(
        tmp_path,
        text="Acme supplies Gamma; Gamma represents 30% of Acme revenue in FY2025.",
        record={
            "related_entity": "Gamma",
            "revenue_dependency_pct": 30,
            "effective_period": "FY2025",
        },
    )
    frame = compute_graph_features(
        ["2026-01-03T00:00:00Z"],
        build_graph_history([first, second], ENTITIES),
        entity_ids=["Acme"],
    )
    assert frame.iloc[0]["graph.quality.revenue.mixed_weight_scope"] == 1
    assert np.isnan(frame.iloc[0]["graph.dependency.revenue.known_weight_sum"])


def test_late_review_of_old_relation_cannot_reopen_newer_termination(tmp_path):
    old, _ = make_fact(tmp_path, reviewed="2026-01-10T10:00:00Z")
    assert graph_asof(
        build_graph_history([old], ENTITIES), "2026-01-11T00:00:00Z", max_age="2D"
    ).edges.empty
    stopped, _ = make_fact(
        tmp_path,
        text="Acme terminated its supply relationship with Beta on 2026-01-05.",
        record={"revenue_dependency_pct": None, "effective_status": "terminated"},
        received="2026-01-05T10:00:00Z",
        processed="2026-01-05T10:01:00Z",
        reviewed="2026-01-05T10:02:00Z",
        effective_from="2026-01-05",
    )
    assert graph_asof(
        build_graph_history([old, stopped], ENTITIES), "2026-01-11T00:00:00Z"
    ).edges.empty


def test_late_review_of_old_guidance_does_not_replace_newer_source_statement(tmp_path):
    fields = {
        "company": "Acme",
        "metric": "revenue",
        "statement_type": "management_guidance",
        "currency": "USD",
        "unit_scale": "billion",
        "fiscal_period_label": "FY2026",
    }
    old, _ = make_fact(
        tmp_path,
        "C",
        text="Acme guidance revenue USD 8 billion in FY2026.",
        record={**fields, "value": 8},
        reviewed="2026-01-10T10:00:00Z",
    )
    current, _ = make_fact(
        tmp_path,
        "C",
        text="Acme guidance revenue USD 9 billion in FY2026.",
        record={**fields, "value": 9},
        received="2026-01-05T10:00:00Z",
        processed="2026-01-05T10:01:00Z",
        reviewed="2026-01-05T10:02:00Z",
    )
    specs = {
        "guidance": {
            "metric": "revenue",
            "statement_type": "management_guidance",
            "currency": "USD",
            "unit_scale": "billion",
            "segment_name": None,
        }
    }
    values = compute_statement_features(
        ["2026-01-11T00:00:00Z"],
        statements_frame([old, current], ENTITIES),
        entity_ids=["Acme"],
        specifications=specs,
    )
    assert values.iloc[0]["statements.guidance.value"] == 9
