"""Discovery CLI integration on synthetic snapshots and fake transports only."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from pfe.data import storage as storage_module
from pfe.data.qualification import DiscoveryError, DiscoveryResponse
from pfe.data.storage import RawStore


@pytest.fixture
def qualification_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qualify_lse.py"
    spec = importlib.util.spec_from_file_location("qualify_lse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot_file(tmp_path):
    path = tmp_path / "synthetic-discovery.json"
    path.write_text(
        json.dumps(
            {
                "provider": "LSE",
                "audited_at": "2026-10-11T12:00:00Z",
                "catalog": [{"symbol": "QQQ", "dataset": "etf", "unit": "USD", "country": "US"}],
                "reference": [{"dataset": "dividends"}],
                "catalog_complete": True,
                "catalog_authoritative": True,
                "reference_complete": True,
                "reference_authoritative": True,
                "usage": {"remaining": 10, "api_key": "synthetic-offline-secret"},
            }
        )
    )
    return path


def _report(store, collection_id):
    return json.loads(store.read_page(collection_id, "qualification"))


def test_offline_without_lse_import_key_network_or_overwrite(
    qualification_cli, snapshot_file, tmp_path, monkeypatch, capsys
):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline qualification reached an online transport")

    monkeypatch.delenv("LSE_API_KEY", raising=False)
    monkeypatch.setattr(qualification_cli, "BoundedLSETransport", no_network)
    monkeypatch.setitem(sys.modules, "lse", None)
    args = [
        "--snapshot",
        str(snapshot_file),
        "--audit-id",
        "offline-test",
        "--symbol",
        "QQQ@etf",
        "--symbol",
        "MISSING",
        "--store-dir",
        str(tmp_path / "raw"),
    ]
    collection_id = qualification_cli.main(args)
    store = RawStore(tmp_path / "raw")
    first = store.manifest(collection_id)
    report = _report(store, collection_id)
    assert len(report["requirements"]) == 34
    assert [row["status"] for row in report["requested_symbols"]] == ["available", "absent"]
    assert first["complete"] is True
    assert qualification_cli.main(args) == collection_id
    assert store.manifest(collection_id) == first
    for path in (tmp_path / "raw").rglob("*"):
        if path.is_file():
            assert b"synthetic-offline-secret" not in path.read_bytes()
    assert "synthetic-offline-secret" not in capsys.readouterr().out


def test_new_audit_id_preserves_previous_snapshot(qualification_cli, snapshot_file, tmp_path):
    args = ["--snapshot", str(snapshot_file), "--store-dir", str(tmp_path / "raw")]
    first_id = qualification_cli.main([*args, "--audit-id", "first"])
    second_id = qualification_cli.main([*args, "--audit-id", "second"])
    assert first_id != second_id
    store = RawStore(tmp_path / "raw")
    assert _report(store, first_id)["audit_id"] == "first"
    assert _report(store, second_id)["audit_id"] == "second"


def test_derived_offline_id_is_stable(qualification_cli, snapshot_file, tmp_path):
    args = ["--snapshot", str(snapshot_file), "--store-dir", str(tmp_path / "raw")]
    assert qualification_cli.main(args) == qualification_cli.main(args)


def test_truncated_snapshot_cannot_complete_manifest_or_prove_absence(
    qualification_cli, snapshot_file, tmp_path
):
    snapshot = json.loads(snapshot_file.read_text())
    snapshot["catalog_complete"] = False
    snapshot["catalog_issues"] = ["provider_truncated"]
    snapshot_file.write_text(json.dumps(snapshot))
    args = [
        "--snapshot",
        str(snapshot_file),
        "--store-dir",
        str(tmp_path / "raw"),
        "--audit-id",
        "partial",
        "--symbol",
        "MISSING",
    ]
    collection_id = qualification_cli.main(args)
    store = RawStore(tmp_path / "raw")
    assert store.manifest(collection_id)["complete"] is False
    assert _report(store, collection_id)["requested_symbols"][0]["status"] == "unqualified"
    original = store.manifest(collection_id)
    qualification_cli.main(args)
    assert store.manifest(collection_id) == original


def test_offline_row_cap_overrides_asserted_completeness(
    qualification_cli, snapshot_file, tmp_path
):
    collection_id = qualification_cli.main(
        [
            "--snapshot",
            str(snapshot_file),
            "--max-rows",
            "1",
            "--symbol",
            "MISSING",
            "--store-dir",
            str(tmp_path / "raw"),
        ]
    )
    report = _report(RawStore(tmp_path / "raw"), collection_id)
    assert report["catalogue_coverage"]["complete"] is False
    assert report["requested_symbols"][0]["status"] == "unqualified"


def test_reference_only_cap_prevents_manifest_completion(
    qualification_cli, snapshot_file, tmp_path
):
    snapshot = json.loads(snapshot_file.read_text())
    snapshot["catalog"] = []
    snapshot["reference"] = [{"dataset": "dividends"}, {"dataset": "stock_splits"}]
    snapshot_file.write_text(json.dumps(snapshot))
    collection_id = qualification_cli.main(
        ["--snapshot", str(snapshot_file), "--max-rows", "2", "--store-dir", str(tmp_path / "raw")]
    )
    store = RawStore(tmp_path / "raw")
    report = _report(store, collection_id)
    assert report["catalogue_coverage"]["complete"]
    assert not report["reference_coverage"]["complete"]
    assert not store.manifest(collection_id)["complete"]


@pytest.mark.parametrize(
    "argument,value",
    [
        ("--max-rows", "0"),
        ("--max-bytes", "1"),
        ("--max-requests", "0"),
        ("--symbol", "QQQ@"),
        ("--symbol", "QQQ@etf@options"),
        ("--audit-id", "../invalid"),
    ],
)
def test_cli_invalid_or_capped_inputs_never_call_transport(
    qualification_cli, snapshot_file, tmp_path, monkeypatch, argument, value
):
    def no_network(*args, **kwargs):
        raise AssertionError("Invalid input reached transport")

    monkeypatch.setattr(qualification_cli, "BoundedLSETransport", no_network)
    with pytest.raises(SystemExit) as error:
        qualification_cli.main(
            [
                "--snapshot",
                str(snapshot_file),
                "--store-dir",
                str(tmp_path / "raw"),
                argument,
                value,
            ]
        )
    assert error.value.code == 2


def test_explicit_online_execution_requires_audit_id_and_key(
    qualification_cli, tmp_path, monkeypatch
):
    monkeypatch.delenv("LSE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        qualification_cli.main(["--run", "--store-dir", str(tmp_path / "raw")])
    with pytest.raises(SystemExit):
        qualification_cli.main(
            ["--run", "--audit-id", "explicit", "--store-dir", str(tmp_path / "raw")]
        )
    with pytest.raises(SystemExit):
        qualification_cli.main(["--audit-id", "explicit", "--store-dir", str(tmp_path / "raw")])


def test_online_failed_endpoint_resumes_without_refetch_or_secret_in_store(
    qualification_cli, tmp_path, monkeypatch, capsys
):
    responses = {
        "/meta": {},
        "/catalog": [{"symbol": "QQQ", "dataset": "etf", "country": "US"}],
        "/reference": [],
        "/usage": {"remaining": 1},
    }
    calls, fail = [], True

    class FakeTransport:
        def __init__(self, key):
            assert key == "synthetic-online-secret"

        def get(self, endpoint, *, max_bytes):
            calls.append(endpoint)
            if endpoint == "/reference" and fail:
                raise DiscoveryError("Synthetic metadata failure")
            payload = responses[endpoint]
            return DiscoveryResponse(payload, len(json.dumps(payload).encode()))

    monkeypatch.setattr(qualification_cli, "BoundedLSETransport", FakeTransport)
    monkeypatch.setenv("LSE_API_KEY", "synthetic-online-secret")
    args = ["--run", "--audit-id", "resume", "--store-dir", str(tmp_path / "raw")]
    with pytest.raises(SystemExit):
        qualification_cli.main(args)
    assert calls == ["/meta", "/catalog", "/reference"]
    fail = False
    collection_id = qualification_cli.main(args)
    assert calls == ["/meta", "/catalog", "/reference", "/reference", "/usage"]
    monkeypatch.delenv("LSE_API_KEY")
    qualification_cli.main(args)  # Finished immutable audit needs neither key nor network.
    assert len(calls) == 5
    store = RawStore(tmp_path / "raw")
    assert store.manifest(collection_id)["complete"]
    assert {page["page_id"] for page in store.manifest(collection_id)["pages"]} == {
        "meta",
        "catalog",
        "reference",
        "usage",
        "snapshot",
        "qualification",
    }
    output = capsys.readouterr()
    assert "synthetic-online-secret" not in output.out + output.err
    assert "synthetic-online-secret" not in json.dumps(store.manifest(collection_id))


def test_qualification_write_failure_resumes_saved_snapshot_without_transport(
    qualification_cli, snapshot_file, tmp_path, monkeypatch
):
    original_write = RawStore.write_page
    failed = False

    def fail_report_once(self, collection_id, page_id, payload, **kwargs):
        nonlocal failed
        if page_id == "qualification" and not failed:
            failed = True
            raise OSError("Synthetic interrupted report")
        return original_write(self, collection_id, page_id, payload, **kwargs)

    monkeypatch.setattr(RawStore, "write_page", fail_report_once)
    args = [
        "--snapshot",
        str(snapshot_file),
        "--audit-id",
        "resume-offline",
        "--store-dir",
        str(tmp_path / "raw"),
    ]
    with pytest.raises(SystemExit):
        qualification_cli.main(args)
    collection_id = qualification_cli.main(args)
    store = RawStore(tmp_path / "raw")
    assert store.manifest(collection_id)["complete"]
    assert len(store.manifest(collection_id)["pages"]) == 2


def test_interrupted_endpoint_manifest_commit_recovers_original_page(
    qualification_cli, tmp_path, monkeypatch
):
    calls = []
    responses = {
        "/meta": {},
        "/catalog": [{"symbol": "QQQ", "dataset": "etf"}],
        "/reference": [],
        "/usage": {"remaining": 1},
    }

    class FakeTransport:
        def __init__(self, key):
            pass

        def get(self, endpoint, *, max_bytes):
            calls.append(endpoint)
            payload = responses[endpoint]
            return DiscoveryResponse(payload, len(json.dumps(payload).encode()))

    publish = storage_module._atomic_write
    commits = 0

    def interrupted_publish(path, body, *, replace):
        nonlocal commits
        if path.name == "manifest.json" and replace:
            commits += 1
            if commits == 2:  # Catalogue raw/journal written; manifest not yet committed.
                raise OSError("Synthetic interrupted manifest")
        return publish(path, body, replace=replace)

    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(qualification_cli, "BoundedLSETransport", FakeTransport)
    monkeypatch.setattr(storage_module, "_atomic_write", interrupted_publish)
    args = ["--run", "--audit-id", "recover-endpoint", "--store-dir", str(tmp_path / "raw")]
    with pytest.raises(SystemExit):
        qualification_cli.main(args)
    assert calls == ["/meta", "/catalog"]
    collection_id = qualification_cli.main(args)
    assert calls == ["/meta", "/catalog", "/reference", "/usage"]
    store = RawStore(tmp_path / "raw")
    assert store.manifest(collection_id)["complete"]
    page = json.loads(store.read_page(collection_id, "catalog"))
    snapshot = json.loads(store.read_page(collection_id, "snapshot"))
    assert page["captured_at"] == snapshot["endpoint_captured_at"]["catalog"]
