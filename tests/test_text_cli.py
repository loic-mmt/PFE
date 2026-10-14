import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from pfe import DocumentStore, Extractor, evaluate_extractions, extraction_schema
from pfe.extraction.schemas import ENVELOPES


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "text_pipeline.py"
    spec = importlib.util.spec_from_file_location("text_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_offline_cli_import_extract_review_build_parquet(tmp_path, monkeypatch, capsys):
    cli = cli_module()
    import pfe.data.documents as document_module
    import pfe.extraction.engine as engine_module

    monkeypatch.setattr(document_module, "utc_now", lambda: "2026-01-02T10:00:00Z")
    monkeypatch.setattr(engine_module, "utc_now", lambda: "2026-01-02T10:01:00Z")
    source = tmp_path / "report.txt"
    source.write_text("Acme supplies Beta; Beta represents 20% of Acme revenue.")
    archive = str(tmp_path / "documents")
    index = str(tmp_path / "index.json")
    assert (
        cli.main(
            [
                "import",
                str(source),
                "--logical-id",
                "report",
                "--document-archive",
                archive,
                "--index",
                index,
            ]
        )
        == 0
    )
    original = json.loads(Path(index).read_text())
    assert (
        cli.main(
            [
                "import",
                str(source),
                "--logical-id",
                "report",
                "--document-archive",
                archive,
                "--index",
                index,
            ]
        )
        == 0
    )
    assert original == json.loads(Path(index).read_text())
    doc = DocumentStore(archive).load(original[0]["collection_id"])
    props = extraction_schema("A")["properties"]["relationships"]["items"]["properties"]
    record = dict.fromkeys(props)
    record.update(
        source_company="Acme",
        related_entity="Beta",
        relation_type="supplier",
        certainty="confirmed",
        effective_status="existing",
        revenue_dependency_pct=20,
        source_document_id=doc.source_document_id,
        evidence_quote=doc.text,
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps({"relationships": [record]}))
    facts = str(tmp_path / "facts.json")
    extract_args = [
        "extract",
        "--index",
        index,
        "--document-archive",
        archive,
        "--extraction-archive",
        str(tmp_path / "extraction"),
        "--tasks",
        "A",
        "--response-json",
        str(response),
        "--output",
        facts,
    ]
    assert cli.main(extract_args) == 0 and cli.main(extract_args) == 0
    records = json.loads(Path(facts).read_text())
    assert len(records) == 1
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            [
                {
                    "fact_id": records[0]["fact_id"],
                    "approved": True,
                    "reviewer": "fixture",
                    "reviewed_at": "2026-01-02T10:02:00Z",
                    "notes": "Verified synthetic supplier direction and 20% revenue share.",
                }
            ]
        )
    )
    review_args = [
        "review",
        "--facts",
        facts,
        "--reviews",
        str(reviews),
        "--index",
        index,
        "--document-archive",
        archive,
        "--output",
        facts,
    ]
    assert cli.main(review_args) == 0 and cli.main(review_args) == 0
    assert len(json.loads(Path(facts).read_text())) == 2
    entities = tmp_path / "entities.json"
    entities.write_text(
        json.dumps(
            [
                {
                    "alias": alias,
                    "entity_id": alias,
                    "node_type": "company",
                    "available_at": "2026-01-01T00:00:00Z",
                }
                for alias in ("Acme", "Beta")
            ]
        )
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(["2026-01-02T10:01:00Z", "2026-01-03T00:00:00Z"]))
    root = tmp_path / "outputs"
    assert (
        cli.main(
            [
                "build",
                "--facts",
                facts,
                "--entities",
                str(entities),
                "--decisions",
                str(decisions),
                "--entity-id",
                "Acme",
                "--output-dir",
                str(root),
            ]
        )
        == 0
    )
    values = pd.read_parquet(root / "graph/node_features.parquet")
    assert values.iloc[0]["graph.degree.supplier.out"] == 0
    assert values.iloc[1]["graph.degree.supplier.out"] == 1
    assert (root / "events/events_structured.parquet").exists()
    assert (root / "fundamentals/management_guidance.parquet").exists()
    assert sum(path.stat().st_size for path in root.rglob("*.*")) < 1_048_576
    capsys.readouterr()


def test_annotated_corpus_exact_precision_recall_mechanics(tmp_path):
    corpus = json.loads((Path(__file__).parent / "fixtures/extraction_corpus.json").read_text())
    store, engine = DocumentStore(tmp_path / "documents"), Extractor(tmp_path / "extraction")
    predicted, expected = [], []
    for case in corpus["cases"]:
        doc = store.import_bytes(
            case["text"].encode(), logical_id=case["case_id"], received_at="2026-01-02T10:00:00Z"
        )
        records = [
            {**record, "source_document_id": doc.source_document_id} for record in case["response"]
        ]
        result = engine.import_response(
            doc,
            case["task"],
            {ENVELOPES[case["task"]]: records},
            received_at="2026-01-02T10:01:00Z",
        )
        predicted.extend(
            {"case_id": case["case_id"], "task": case["task"], "record": fact["record"]}
            for fact in result.accepted
        )
        expected.extend(
            {
                "case_id": case["case_id"],
                "task": case["task"],
                "record": {**record, "source_document_id": doc.source_document_id},
            }
            for record in case["expected"]
        )
    report = evaluate_extractions(predicted, expected, minimum_precision=1, minimum_recall=1)
    assert report["passed"] and report["true_positives"] == 11
    assert report["false_positives"] == report["false_negatives"] == 0


def test_evaluation_cli_fails_unqualified_empty_corpus(tmp_path, capsys):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    output = tmp_path / "report.json"
    assert (
        cli_module().main(
            [
                "evaluate",
                "--predicted",
                str(path),
                "--expected",
                str(path),
                "--minimum-precision",
                ".99",
                "--minimum-recall",
                ".9",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert json.loads(output.read_text())["passed"] is False
    capsys.readouterr()


def test_cli_rejects_large_json_and_preserves_existing_index(tmp_path):
    cli = cli_module()
    path = tmp_path / "large.json"
    path.write_bytes(b"x" * 20)
    with pytest.raises(ValueError, match="byte limit"):
        cli.read_json(path, max_bytes=10)
    path = tmp_path / "index.json"
    cli.append_json(path, [{"id": "one", "value": 1}], key=lambda row: row["id"])
    with pytest.raises(ValueError, match="Conflicting"):
        cli.append_json(path, [{"id": "one", "value": 2}], key=lambda row: row["id"])
    assert json.loads(path.read_text()) == [{"id": "one", "value": 1}]
