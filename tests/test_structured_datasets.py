"""Graph/event model inputs retain knowledge gates and compatible evidence."""

import numpy as np
import pandas as pd
import pytest
from test_graph_events import ENTITIES, make_fact

from pfe import build_event_history, build_graph_history
from pfe.datasets import (
    DatasetLimits,
    build_event_tensor,
    build_feature_panel,
    iter_graph_tensors,
    load_model_dataset,
    save_model_dataset,
)


def panel(decisions):
    rows = [
        dict(
            entity_id=entity,
            metric="momentum",
            unit="fraction",
            observation_date="2026-01-01",
            published_at=None,
            available_at="2026-01-01T00:00:00Z",
            value=0.1,
            source_id="fixture",
            revision_id="1",
            schema_version="1",
        )
        for entity in ("Acme", "Beta")
    ]
    return build_feature_panel(
        decisions,
        pd.DataFrame(rows),
        entity_ids=["Acme", "Beta"],
        registry=[
            dict(
                name="momentum",
                metric="momentum",
                unit="fraction",
                source_id="fixture",
                window="5D",
                dependencies=["returns"],
            )
        ],
    )


def test_graph_edge_only_after_review_with_direction_weight_kind_and_owner(tmp_path):
    fact, _ = make_fact(tmp_path)
    graph = build_graph_history([fact], ENTITIES)
    inputs = panel(["2026-01-02T10:01:59Z", "2026-01-02T10:02:00Z"])
    before, after = list(iter_graph_tensors(inputs, graph))
    assert before.edge_index.shape == (2, 0)
    assert after.node_ids == ("Acme", "Beta")
    np.testing.assert_array_equal(after.edge_index, [[0], [1]])
    assert after.edge_weight.tolist() == [0.2]
    assert after.edge_weight_kind == ("revenue_dependency_pct",)
    assert after.edge_owner_entity_ids == ("Acme",)
    assert after.edge_source_documents == (fact["source_document_id"],)
    save_model_dataset(tmp_path / "dataset", inputs, graphs=[before, after])
    arrays, manifest = load_model_dataset(tmp_path / "dataset")
    assert manifest["graphs"][1]["edge_weight_kind"] == ["revenue_dependency_pct"]
    assert arrays["graph_1_edge_index"].shape == (2, 1)


def test_event_sequence_padding_link_and_model_clocks(tmp_path):
    fact, doc = make_fact(tmp_path, task="D", text="Acme launched a product.")
    history = build_event_history([fact], ENTITIES)
    outputs = [
        dict(
            status="accepted",
            is_model_output=True,
            kind="embedding",
            source_document_id=doc.source_document_id,
            text_sha256=doc.text_sha256,
            model={"model_revision": "digest:fixture"},
            available_at="2026-01-02T10:03:00Z",
            value={"embedding": [0.1, 0.2]},
        )
    ]
    decisions = ["2026-01-02T10:01:59Z", "2026-01-02T10:02:00Z", "2026-01-02T10:03:00Z"]
    events = build_event_tensor(
        decisions,
        history,
        entity_ids=["Acme", "Beta"],
        max_events=2,
        enrichments=outputs,
        model_revision="digest:fixture",
        model_available_at="2025-01-01T00:00:00Z",
        embedding_dimensions=2,
    )
    assert not events.mask[0].any()
    assert events.mask[1, 0].sum() == 1 and not events.mask[:, 1].any()
    assert not events.embedding_mask[1].any() and events.embedding_mask[2, 0, 0]
    np.testing.assert_array_equal(events.embeddings[2, 0, 0], [0.1, 0.2])
    assert events.age_seconds[1, 0, 0] == 0
    assert events.mask.shape == (3, 2, 2)
    save_model_dataset(tmp_path / "events", panel(decisions), events=events)
    arrays, _ = load_model_dataset(tmp_path / "events")
    assert arrays["events_embeddings"].shape == (3, 2, 2, 2)


def test_event_embedding_unknown_model_availability_refused(tmp_path):
    fact, _ = make_fact(tmp_path, task="D", text="Acme launched a product.")
    history = build_event_history([fact], ENTITIES)
    with pytest.raises(ValueError, match="model revision"):
        build_event_tensor(["2026-01-02T11:00:00Z"], history, entity_ids=["Acme"], enrichments=[{}])


def test_future_model_release_does_not_backfill_vector(tmp_path):
    fact, doc = make_fact(tmp_path, task="D", text="Acme launched a product.")
    history = build_event_history([fact], ENTITIES)
    output = dict(
        status="accepted",
        is_model_output=True,
        kind="embedding",
        source_document_id=doc.source_document_id,
        text_sha256=doc.text_sha256,
        model={"model_revision": "future"},
        available_at="2026-01-02T10:03:00Z",
        value={"embedding": [1.0]},
    )
    tensor = build_event_tensor(
        ["2026-01-02T11:00:00Z"],
        history,
        entity_ids=["Acme"],
        enrichments=[output],
        model_revision="future",
        model_available_at="2026-01-03T00:00:00Z",
        embedding_dimensions=1,
    )
    assert tensor.mask.any() and not tensor.embedding_mask.any()


def test_event_memory_guard_before_allocation(tmp_path):
    fact, _ = make_fact(tmp_path, task="D", text="Acme launched a product.")
    with pytest.raises(ValueError, match="allocation"):
        build_event_tensor(
            ["2026-01-02T11:00:00Z"],
            build_event_history([fact], ENTITIES),
            entity_ids=["Acme"],
            limits=DatasetLimits(max_bytes=100),
        )


def test_export_rejects_foreign_graphs_and_changed_node_features(tmp_path):
    from dataclasses import replace

    fact, _ = make_fact(tmp_path)
    history = build_graph_history([fact], ENTITIES)
    inputs = panel(["2026-01-02T11:00:00Z"])
    graph = next(iter_graph_tensors(inputs, history))
    changed = replace(inputs, values=inputs.values + 10)
    foreign = next(iter_graph_tensors(changed, history))
    with pytest.raises(ValueError, match="provenance"):
        save_model_dataset(tmp_path / "foreign", inputs, graphs=[foreign])
    with pytest.raises(ValueError, match="identities"):
        save_model_dataset(
            tmp_path / "identities",
            inputs,
            graphs=[replace(graph, node_ids=("wrong-company", "Beta"))],
        )
    with pytest.raises(ValueError, match="feature values"):
        save_model_dataset(tmp_path / "values", inputs, graphs=[replace(graph, x=graph.x + 1)])
    assert not (tmp_path / "foreign").exists()


@pytest.mark.parametrize("mismatch", ["entities", "times"])
def test_export_rejects_event_axes_unaligned_with_panel(tmp_path, mismatch):
    fact, _ = make_fact(tmp_path, task="D", text="Acme launched a product.")
    decisions = ["2026-01-02T11:00:00Z"]
    events = build_event_tensor(
        ["2026-01-03T11:00:00Z"] if mismatch == "times" else decisions,
        build_event_history([fact], ENTITIES),
        entity_ids=["Beta", "Acme"] if mismatch == "entities" else ["Acme", "Beta"],
        max_events=1,
    )
    with pytest.raises(ValueError, match="axes"):
        save_model_dataset(tmp_path / "events", panel(decisions), events=events)
