import io
import json
import sys
from types import SimpleNamespace

import pytest

from pfe import DocumentLimits, DocumentStore, HTTPResponse, documents_frame, parse_document

RECEIPT = "2026-01-02T12:00:00Z"


def test_text_archive_replay_modified_history_and_metadata(tmp_path):
    store = DocumentStore(tmp_path)
    original = store.import_bytes(b"Acme revenue 20.", logical_id="report", received_at=RECEIPT)
    replay = store.import_bytes(
        b"Acme revenue 20.", logical_id="report", received_at="2026-01-03T12:00:00Z"
    )
    changed = store.import_bytes(b"Acme revenue 21.", logical_id="report", received_at=RECEIPT)
    assert original == replay == store.load(original.collection_id)
    assert changed.source_document_id != original.source_document_id
    assert changed.revision_id != original.revision_id
    assert original.available_at == RECEIPT
    assert original.published_at is None
    assert "text" not in documents_frame([original])
    assert documents_frame([original]).point_in_time.iloc[0]


def test_html_excludes_scripts_and_preserves_table_cells():
    text, pages = parse_document(
        b"<script>invent guidance</script><p>Acme &amp; Beta</p>"
        b"<table><tr><td>Revenue</td><td>20</td></tr></table>",
        "text/html",
    )
    assert "invent" not in text and "Acme & Beta" in text
    assert "Revenue\t20" in text and pages == ()


@pytest.mark.parametrize(
    "kind,body", [("text/plain", b""), ("application/zip", b"zip"), ("text/plain", b"\xff")]
)
def test_reject_empty_unsupported_or_invalid_encoding(kind, body):
    with pytest.raises(ValueError):
        parse_document(body, kind)


def test_pdf_text_pages_optional_no_ocr(monkeypatch):
    fake = SimpleNamespace(
        PdfReader=lambda _: SimpleNamespace(
            is_encrypted=False,
            pages=[
                SimpleNamespace(extract_text=lambda: "page one"),
                SimpleNamespace(extract_text=lambda: "page two"),
            ],
        )
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    text, pages = parse_document(b"%PDF-fixture", "application/pdf")
    assert text == "page one\npage two"
    assert pages[1] == {"page": 2, "start": 9, "end": 17}
    with pytest.raises(ValueError, match="page limit"):
        parse_document(b"pdf", "application/pdf", limits=DocumentLimits(max_pdf_pages=1))
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(ImportError, match="optional pypdf"):
        parse_document(b"pdf", "application/pdf")


def test_small_file_and_total_budgets_fail_before_archive(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 100)
    store = DocumentStore(tmp_path / "raw", limits=DocumentLimits(max_document_bytes=10))
    with pytest.raises(ValueError, match="byte limit"):
        store.import_file(path, logical_id="large")
    assert list(store.raw.root.iterdir()) == []
    store = DocumentStore(tmp_path / "bounded", limits=DocumentLimits(max_total_bytes=10))
    with pytest.raises(ValueError, match="budget"):
        store.import_bytes(b"small", logical_id="small")
    assert list(store.raw.root.iterdir()) == []
    with pytest.raises(ValueError, match="text limit"):
        parse_document(b"abcdef", "text/plain", limits=DocumentLimits(max_text_bytes=5))


@pytest.mark.parametrize("value", [True, 0, -1, 1.2])
def test_invalid_document_limits(value):
    with pytest.raises(ValueError):
        DocumentLimits(max_documents=value)


def test_sec_url_exact_no_acceptance_backdating(tmp_path):
    calls = []

    class Transport:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return HTTPResponse(b"<p>Acme reported revenue.</p>", RECEIPT, "text/html")

    doc = DocumentStore(tmp_path).collect_sec(
        cik="0000012345",
        accession="0000012345-26-000001",
        primary_document="report.htm",
        user_agent="Research me@example.org",
        transport=Transport(),
        accepted_at="2026-01-02T11:00:00Z",
    )
    assert (
        calls[0][0] == "https://www.sec.gov/Archives/edgar/data/12345/000001234526000001/report.htm"
    )
    assert calls[0][1]["max_bytes"] == 524_288
    assert doc.published_at is None and doc.available_at == RECEIPT
    assert doc.accepted_at == "2026-01-02T11:00:00Z"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/report",
        "https://other.org/report",
        "https://user:pass@example.org/report",
        "https://example.org/report#fragment",
    ],
)
def test_explicit_urls_are_bounded_and_allowlisted(tmp_path, url):
    with pytest.raises(ValueError):
        DocumentStore(tmp_path).collect_url(url, allowed_hosts=["example.org"], logical_id="report")


@pytest.mark.parametrize(
    "field,value",
    [
        ("cik", "../3"),
        ("accession", "invalid"),
        ("primary_document", "../report.htm"),
        ("user_agent", "anonymous"),
    ],
)
def test_sec_rejects_bad_identifiers_before_transport(tmp_path, field, value):
    args = dict(
        cik=123,
        accession="0000000123-26-000001",
        primary_document="report.htm",
        user_agent="Research me@example.org",
    )
    args[field] = value
    with pytest.raises(ValueError):
        DocumentStore(tmp_path).collect_sec(**args)


def test_future_publication_rejected_and_raw_hash_checked(tmp_path):
    store = DocumentStore(tmp_path)
    with pytest.raises(ValueError, match="publication|published_at"):
        store.import_bytes(
            b"report", logical_id="report", received_at=RECEIPT, published_at="2026-02-01T00:00:00Z"
        )
    doc = store.import_bytes(b"report", logical_id="report", received_at=RECEIPT)
    manifest = store.raw.manifest(doc.collection_id)
    assert json.loads(store.raw.read_page(doc.collection_id, "document"))["text"] == "report"
    assert manifest["complete"]
    assert isinstance(io.BytesIO(store.raw.read_page(doc.collection_id, "raw")).getvalue(), bytes)
