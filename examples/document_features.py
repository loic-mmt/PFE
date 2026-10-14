"""Offline Lot 3 demonstration: synthetic evidence, explicit review, no downloads."""

from tempfile import TemporaryDirectory

from pfe import (
    DocumentStore,
    Extractor,
    adjudicate_fact,
    build_event_history,
    build_graph_history,
    compute_event_features,
    compute_graph_features,
    extraction_schema,
)


def record(task, document, **fields):
    schema = extraction_schema(task)
    envelope = next(iter(schema["properties"]))
    values = dict.fromkeys(schema["properties"][envelope]["items"]["properties"])
    values.update(source_document_id=document.source_document_id, **fields)
    return {envelope: [values]}


def main():
    with TemporaryDirectory(prefix="pfe-document-example-") as root:
        doc = DocumentStore(root + "/documents").import_bytes(
            b"Acme supplies Beta; Beta represents 20% of Acme revenue. "
            b"Acme announced a chip product launch.",
            logical_id="synthetic-report",
            received_at="2026-01-02T10:00:00Z",
            published_at="2026-01-02T09:00:00Z",
        )
        engine = Extractor(root + "/extraction")
        payloads = {
            "A": record(
                "A",
                doc,
                source_company="Acme",
                related_entity="Beta",
                relation_type="supplier",
                certainty="confirmed",
                effective_status="existing",
                revenue_dependency_pct=20,
                evidence_quote="Acme supplies Beta; Beta represents 20% of Acme revenue.",
            ),
            "D": record(
                "D",
                doc,
                event_type="product_launch",
                entities_involved=["Acme"],
                event_status="announced_plan",
                evidence_quote="Acme announced a chip product launch.",
            ),
        }
        facts = []
        for task, payload in payloads.items():
            result = engine.import_response(doc, task, payload, received_at="2026-01-02T10:01:00Z")
            if result.rejected:
                raise AssertionError(result.rejected)
            facts.extend(
                adjudicate_fact(
                    fact,
                    approved=True,
                    reviewer="synthetic-fixture-author",
                    reviewed_at="2026-01-02T10:02:00Z",
                    notes="Synthetic annotation: fields and direction checked.",
                    document=doc,
                )
                for fact in result.accepted
            )
        entities = [
            {
                "alias": alias,
                "entity_id": entity,
                "node_type": "company",
                "available_at": "2026-01-01T00:00:00Z",
            }
            for alias, entity in (("Acme", "ACME"), ("Beta", "BETA"))
        ]
        decisions = ["2026-01-02T10:00:00Z", "2026-01-02T10:03:00Z"]
        graph = build_graph_history(facts, entities)
        events = build_event_history(facts, entities)
        graph_values = compute_graph_features(decisions, graph, entity_ids=["ACME"])
        event_values = compute_event_features(decisions, events, entity_ids=["ACME"])
        print(
            graph_values[["graph.degree.supplier.out", "graph.dependency.revenue.known_weight_sum"]]
        )
        print(event_values[["events.count.product_launch.announced_plan.7D", "events.age_seconds"]])
        assert graph_values.iloc[0]["graph.degree.supplier.out"] == 0
        assert graph_values.iloc[1]["graph.degree.supplier.out"] == 1
        assert event_values.iloc[1]["events.count.product_launch.announced_plan.7D"] == 1


if __name__ == "__main__":
    main()
