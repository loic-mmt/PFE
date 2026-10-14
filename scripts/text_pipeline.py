"""Small-document Lot 3 CLI. No model download, crawling or bulk SEC collection."""

import argparse
import json
from pathlib import Path

import pandas as pd

from pfe import (
    DocumentStore,
    EntityRegistry,
    ExtractionLimits,
    Extractor,
    OllamaBackend,
    adjudicate_fact,
    build_event_history,
    build_graph_history,
    compute_event_features,
    compute_graph_features,
    documents_frame,
    evaluate_extractions,
    merge_graph_declarations,
    statements_frame,
)
from pfe.data.documents import utc_now
from pfe.data.storage import _atomic_write, _json_bytes, _locked


def read_json(path, *, max_bytes=4_194_304):
    with Path(path).open("rb") as stream:
        body = stream.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError("JSON input byte limit exceeded")
    from pfe.extraction.validation import strict_json

    return strict_json(body.decode("utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _json_bytes(value)
    if len(body) > 4_194_304:
        raise ValueError("JSON output exceeds 4 MiB CLI budget")
    _atomic_write(path, body, replace=True)


def append_json(path, values, *, key):
    """Keep every prior version, reject conflicting immutable identities."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path.parent):
        prior = read_json(path) if path.exists() else []
        result = {key(item): item for item in prior}
        for item in values:
            identity = key(item)
            if identity in result and result[identity] != item:
                raise ValueError("Conflicting immutable index entry")
            result[identity] = item
        write_json(path, list(result.values()))
    return list(result.values())


def parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write once atomically; no partial public Parquet on interruption.
    import io

    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    body = buffer.getvalue()
    if len(body) > 8_388_608:
        raise ValueError("Parquet output exceeds 8 MiB CLI budget")
    _atomic_write(path, body, replace=True)


def load_documents(args):
    records = read_json(args.index)
    store = DocumentStore(args.document_archive)
    if not isinstance(records, list):
        raise ValueError("Document index must be a JSON list")
    if args.document_id:
        records = [record for record in records if record["source_document_id"] == args.document_id]
    if len(records) > store.limits.max_documents:
        raise ValueError("Document index exceeds per-run document limit; use an explicit subset")
    docs = [store.load(record["collection_id"]) for record in records]
    if args.document_id:
        docs = [doc for doc in docs if doc.source_document_id == args.document_id]
    if not docs:
        raise ValueError("No selected archived document")
    return docs


def make_backend(args):
    return (
        OllamaBackend(
            args.model,
            model_revision=args.model_revision,
            options={"num_predict": args.num_predict},
        )
        if args.model
        else None
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("import", "url", "sec"):
        command = commands.add_parser(name)
        command.add_argument("--document-archive", default="data/raw/documents")
        command.add_argument("--index", default="data/metadata/documents.json")
        command.add_argument("--logical-id", required=name != "sec")
        command.add_argument("--source-id", default="sec" if name == "sec" else "local")
        command.add_argument("--rights", default="unknown")
        command.add_argument("--published-at")
        command.add_argument("--accepted-at")
        command.add_argument("--entity-id")
        command.add_argument("--observation-date")
        if name == "import":
            command.add_argument("path")
        elif name == "url":
            command.add_argument("url")
            command.add_argument("--allowed-host", action="append", required=True)
        else:
            command.add_argument("--cik", required=True)
            command.add_argument("--accession", required=True)
            command.add_argument("--primary-document", required=True)
        if name != "import":
            command.add_argument("--user-agent", required=True)

    for name in ("extract", "enrich", "review"):
        command = commands.add_parser(name)
        command.add_argument("--document-archive", default="data/raw/documents")
        command.add_argument("--index", default="data/metadata/documents.json")
        command.add_argument("--document-id")
        command.add_argument(
            "--output",
            default="data/extraction/facts.json"
            if name != "enrich"
            else "data/events/model_outputs.json",
        )
        if name == "review":
            command.add_argument("--facts", required=True)
            command.add_argument("--reviews", required=True)
        else:
            command.add_argument("--extraction-archive", default="data/raw/extraction")
            command.add_argument("--model")
            command.add_argument("--model-revision")
            command.add_argument("--chunk-chars", type=int, default=12_000)
            command.add_argument("--num-predict", type=int, default=2048)
            if name == "extract":
                command.add_argument(
                    "--tasks", nargs="+", choices=list("ABCD"), default=list("ABCD")
                )
                command.add_argument(
                    "--response-json", help="Offline response envelope for one task/document"
                )
                command.add_argument("--author", default="offline-import")
            else:
                command.add_argument("--kind", choices=("sentiment", "embedding"), required=True)

    build = commands.add_parser("build")
    build.add_argument("--facts", required=True)
    build.add_argument("--entities", required=True, help="Explicit historical aliases JSON")
    build.add_argument("--decisions", required=True, help="JSON list of aware decision timestamps")
    build.add_argument("--entity-id", action="append", required=True)
    build.add_argument("--output-dir", default="data")
    build.add_argument("--event-ids", help="Explicit fact_id->event_id cross-document dedup map")
    build.add_argument(
        "--graph-declarations", help="Qualified structural observations Parquet <=8 MiB"
    )
    build.add_argument("--node-features", help="Small normalized observations Parquet, <=8 MiB")
    build.add_argument("--node-feature-units", help="JSON metric/horizon->unit map")
    build.add_argument("--neighbor-relations", nargs="+", default=("supplier", "customer"))
    build.add_argument("--neighbor-node-type", action="append")
    build.add_argument("--enrichments")
    build.add_argument("--model-revision")
    build.add_argument("--embedding-dimensions", type=int)
    build.add_argument("--statement-specifications")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predicted", required=True)
    evaluate.add_argument("--expected", required=True)
    evaluate.add_argument("--minimum-precision", type=float, required=True)
    evaluate.add_argument("--minimum-recall", type=float, required=True)
    evaluate.add_argument("--output", default="data/extraction/evaluation.json")
    args = parser.parse_args(argv)

    if args.command in ("import", "url", "sec"):
        store = DocumentStore(args.document_archive)
        metadata = {
            name: getattr(args, name)
            for name in (
                "logical_id",
                "source_id",
                "rights",
                "published_at",
                "accepted_at",
                "entity_id",
                "observation_date",
            )
        }
        if metadata["logical_id"] is None:
            del metadata["logical_id"]
        if args.command == "import":
            doc = store.import_file(args.path, **metadata)
        elif args.command == "url":
            doc = store.collect_url(
                args.url, allowed_hosts=args.allowed_host, user_agent=args.user_agent, **metadata
            )
        else:
            doc = store.collect_sec(
                cik=args.cik,
                accession=args.accession,
                primary_document=args.primary_document,
                user_agent=args.user_agent,
                **metadata,
            )
        index = append_json(
            args.index,
            documents_frame([doc]).to_dict("records"),
            key=lambda record: record["source_document_id"],
        )
        parquet(Path(args.index).with_name("source_documents.parquet"), pd.DataFrame(index))
        print(
            json.dumps({"document_id": doc.source_document_id, "collection_id": doc.collection_id})
        )
    elif args.command in ("extract", "enrich"):
        docs = load_documents(args)
        engine = Extractor(
            args.extraction_archive,
            backend=make_backend(args),
            limits=ExtractionLimits(chunk_chars=args.chunk_chars),
        )
        if args.command == "enrich":
            outputs = [engine.enrich(doc, kind=args.kind) for doc in docs]
            append_json(args.output, outputs, key=lambda record: record["enrichment_id"])
            print(json.dumps({"enrichments": len(outputs)}))
        else:
            if args.response_json and (len(docs) != 1 or len(args.tasks) != 1 or args.model):
                raise ValueError("Offline JSON import requires one task/document and no model")
            results = [
                engine.import_response(
                    doc,
                    args.tasks[0],
                    read_json(args.response_json, max_bytes=engine.limits.max_response_bytes),
                    author=args.author,
                )
                if args.response_json
                else engine.extract(doc, tasks=tuple(args.tasks))
                for doc in docs
            ]
            accepted = [fact for result in results for fact in result.accepted]
            append_json(
                args.output,
                accepted,
                key=lambda record: (record["fact_id"], record["available_at"]),
            )
            output = Path(args.output)
            for name, values in (
                ("rejected", [r for result in results for r in result.rejected]),
                ("runs", [r for result in results for r in result.runs]),
            ):
                append_json(
                    output.with_name(name + ".json"), values, key=lambda record: _json_bytes(record)
                )
            print(
                json.dumps(
                    {
                        "accepted": len(accepted),
                        "rejected": sum(len(r.rejected) for r in results),
                        "complete": all(result.complete for result in results),
                    }
                )
            )
            if not all(result.complete for result in results):
                return 2
    elif args.command == "review":
        documents = {doc.source_document_id: doc for doc in load_documents(args)}
        facts = read_json(args.facts)
        base = {fact["fact_id"]: fact for fact in facts if fact["review_status"] == "unreviewed"}
        reviewed = []
        for review in read_json(args.reviews):
            fact = base[review["fact_id"]]
            reviewed.append(
                adjudicate_fact(
                    fact,
                    document=documents[fact["source_document_id"]],
                    approved=review["approved"],
                    reviewer=review["reviewer"],
                    reviewed_at=review.get("reviewed_at") or utc_now(),
                    notes=review["notes"],
                    effective_from=review.get("effective_from"),
                    effective_to=review.get("effective_to"),
                )
            )
        append_json(
            args.output,
            facts + reviewed,
            key=lambda record: (record["fact_id"], record["available_at"]),
        )
        print(json.dumps({"reviews": len(reviewed)}))
    elif args.command == "build":
        facts = read_json(args.facts)
        entities = EntityRegistry(read_json(args.entities))
        decisions = read_json(args.decisions)
        if len(decisions) > 512 or len(args.entity_id) > 32:
            raise ValueError(
                "CLI build limit: 512 decisions and 32 entities; split runs explicitly"
            )
        root = Path(args.output_dir)
        graph = build_graph_history(facts, entities)
        if args.graph_declarations:
            if Path(args.graph_declarations).stat().st_size > 8_388_608:
                raise ValueError("Structural graph input exceeds 8 MiB CLI budget")
            graph = merge_graph_declarations(graph, pd.read_parquet(args.graph_declarations))
        events = build_event_history(
            facts, entities, event_ids=read_json(args.event_ids) if args.event_ids else None
        )
        statements = statements_frame(facts, entities)
        node_features = None
        if args.node_features:
            if Path(args.node_features).stat().st_size > 8_388_608:
                raise ValueError("Node feature input exceeds 8 MiB CLI budget")
            node_features = pd.read_parquet(args.node_features)
        graph_features = compute_graph_features(
            decisions,
            graph,
            entity_ids=args.entity_id,
            node_features=node_features,
            node_feature_units=read_json(args.node_feature_units)
            if args.node_feature_units
            else None,
            neighbor_relations=tuple(args.neighbor_relations),
            neighbor_node_types=tuple(args.neighbor_node_type or ("company",)),
        )
        event_features = compute_event_features(
            decisions,
            events,
            entity_ids=args.entity_id,
            enrichments=read_json(args.enrichments) if args.enrichments else (),
            model_revision=args.model_revision,
            embedding_dimensions=args.embedding_dimensions,
        )
        for relative, frame in (
            ("graph/nodes.parquet", graph.nodes),
            ("graph/edges_history.parquet", graph.edges),
            ("graph/node_features.parquet", graph_features.reset_index()),
            ("events/events_structured.parquet", events.events),
            ("events/event_entity_links.parquet", events.links),
            ("events/event_features.parquet", event_features.reset_index()),
        ):
            parquet(root / relative, frame)
        for kind, filename in (
            ("reported_actual", "company_statements"),
            ("management_guidance", "management_guidance"),
            ("analyst_estimate", "analyst_statements"),
        ):
            parquet(
                root / f"fundamentals/{filename}.parquet",
                statements.loc[statements.statement_type.eq(kind)],
            )
        if args.enrichments:
            records = read_json(args.enrichments)
            parquet(
                root / "events/event_embeddings.parquet",
                pd.DataFrame(
                    [
                        {
                            "source_document_id": item["source_document_id"],
                            "enrichment_id": item["enrichment_id"],
                            "available_at": item["available_at"],
                            "model_json": _json_bytes(item["model"]).decode(),
                            "text_sha256": item["text_sha256"],
                            "value_json": _json_bytes(item["value"]).decode(),
                            "is_model_output": True,
                        }
                        for item in records
                        if item["kind"] == "embedding"
                    ]
                ),
            )
        if args.statement_specifications:
            from pfe import compute_statement_features

            values = compute_statement_features(
                decisions,
                statements,
                entity_ids=args.entity_id,
                specifications=read_json(args.statement_specifications),
            )
            parquet(root / "fundamentals/statement_features.parquet", values.reset_index())
        write_json(
            root / "metadata/graph_event_issues.json",
            {"graph": graph.issues.to_dict("records"), "events": events.issues.to_dict("records")},
        )
        print(
            json.dumps(
                {
                    "edges": len(graph.edges),
                    "events": len(events.events),
                    "semantic_review_required": True,
                }
            )
        )
    else:
        report = evaluate_extractions(
            read_json(args.predicted),
            read_json(args.expected),
            minimum_precision=args.minimum_precision,
            minimum_recall=args.minimum_recall,
        )
        write_json(args.output, report)
        print(json.dumps(report))
        if not report["passed"]:
            return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, ImportError) as error:
        raise SystemExit(str(error)) from None
