"""Raw-store invariants using synthetic JSON and local temporary files only."""

import hashlib
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pfe.data import storage
from pfe.data.storage import CorruptStoreError, ImmutablePageError, RawStore, RawStoreError


def _collection(store, **kwargs):
    return store.begin("synthetic", "daily", {"start": "2026-01-01", "end": "2026-02-01"}, **kwargs)


def _raw_path(root, collection_id, page_id):
    stem = hashlib.sha256(page_id.encode()).hexdigest()
    return Path(root) / collection_id / "pages" / f"{stem}.bin"


def _process_write(root, collection_id, page_id):
    RawStore(root).write_page(collection_id, page_id, {"rows": [page_id]}, row_count=1)


def test_identity_canonical_replay_and_license_revision(tmp_path):
    store = RawStore(tmp_path)
    request = {"end": "2026-02-01", "start": "2026-01-01"}
    metadata = {"revision_id": "r1", "license_notes": "Synthetic fixtures only"}
    collection_id = store.begin("synthetic", "daily", request, source_metadata=metadata)
    original = store.manifest(collection_id)
    assert (
        store.begin(
            "synthetic", "daily", dict(reversed(list(request.items()))), source_metadata=metadata
        )
        == collection_id
    )
    assert store.manifest(collection_id) == original
    request["start"] = "changed"
    metadata["revision_id"] = "r2"
    assert original["request"]["start"] == "2026-01-01"
    assert original["source_metadata"]["revision_id"] == "r1"
    assert _collection(store, source_metadata=metadata) != collection_id
    assert _collection(store, schema_version="2") != _collection(store)


def test_same_window_pages_replayed_without_double_count_and_completion(tmp_path):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    first = store.write_page(
        collection_id,
        "0",
        {"b": 2, "a": 1},
        next_cursor="next",
        row_count=2,
        metadata={"published_at": "2026-01-03T10:00:00Z"},
    )
    assert store.read_page(collection_id, "0") == b'{"a":1,"b":2}'
    assert store.resume_cursor(collection_id) == "next"
    replay = store.write_page(
        collection_id,
        "0",
        {"a": 1, "b": 2},
        next_cursor="next",
        row_count=2,
        metadata={"published_at": "2026-01-03T10:00:00Z"},
    )
    assert replay == first
    assert store.manifest(collection_id)["row_count"] == 2
    store.write_page(collection_id, "1", b"[]", complete=True, row_count=0)
    manifest = store.manifest(collection_id)
    assert manifest["page_count"] == 2
    assert manifest["complete"] is True
    assert store.resume_cursor(collection_id) is None
    assert (
        store.write_page(
            collection_id,
            "0",
            {"a": 1, "b": 2},
            next_cursor="next",
            row_count=2,
            metadata={"published_at": "2026-01-03T10:00:00Z"},
        )
        == first
    )
    assert store.manifest(collection_id) == manifest
    with pytest.raises(ImmutablePageError, match="new snapshot"):
        store.write_page(collection_id, "2", {})


@pytest.mark.parametrize(
    "change",
    [
        {"payload": {"value": 2}},
        {"next_cursor": "changed"},
        {"row_count": 2},
        {"metadata": {"published_at": "2026-01-04T10:00:00Z"}},
        {"complete": True},
    ],
)
def test_changed_body_or_temporal_metadata_rejected(tmp_path, change):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    args = {
        "payload": {"value": 1},
        "row_count": 1,
        "metadata": {"published_at": "2026-01-03T10:00:00Z"},
    }
    store.write_page(collection_id, "0", **args)
    before = store.manifest(collection_id)
    with pytest.raises(ImmutablePageError):
        store.write_page(collection_id, "0", **(args | change))
    assert store.manifest(collection_id) == before


def test_late_revision_preserves_prior_snapshot(tmp_path):
    store = RawStore(tmp_path)
    old_id = store.begin("synthetic", "macro", {"snapshot_at": "2026-01-03T10:00:00Z"})
    new_id = store.begin("synthetic", "macro", {"snapshot_at": "2026-02-03T10:00:00Z"})
    store.write_page(old_id, "0", {"value": 100}, row_count=1, complete=True)
    store.write_page(new_id, "0", {"value": 101}, row_count=1, complete=True)
    assert json.loads(store.read_page(old_id, "0")) == {"value": 100}
    assert json.loads(store.read_page(new_id, "0")) == {"value": 101}


def test_caps_and_unknown_totals_never_imply_completion(tmp_path):
    store = RawStore(tmp_path, max_page_bytes=8)
    collection_id = _collection(store)
    with pytest.raises(RawStoreError, match="max_page_bytes"):
        store.write_page(collection_id, "0", b"too large", complete=True)
    assert store.manifest(collection_id)["page_count"] == 0
    assert store.manifest(collection_id)["complete"] is False
    with pytest.raises(RawStoreError, match="continuing"):
        store.write_page(collection_id, "0", b"[]", complete=True, next_cursor="next")
    with pytest.raises(RawStoreError, match="capped"):
        store.write_page(collection_id, "0", b"[]", complete=True, metadata={"truncated": True})
    store.write_page(collection_id, "0", b"12345678")
    assert store.manifest(collection_id)["complete"] is False
    assert store.manifest(collection_id)["row_count"] is None
    assert store.manifest(collection_id)["known_row_count"] == 0


def test_failed_manifest_commit_recovered_with_original_receipt(tmp_path, monkeypatch):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    replace = storage.os.replace
    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(OSError, match="disk"):
            store.write_page(collection_id, "0", {"rows": [1]}, next_cursor="next", row_count=1)
    assert storage.os.replace is replace
    assert store.manifest(collection_id)["page_count"] == 0
    with pytest.raises(KeyError, match="Unknown page"):
        store.read_page(collection_id, "0")
    with pytest.raises(RawStoreError, match="pending page"):
        store.write_page(collection_id, "1", {})
    record_path = _raw_path(tmp_path, collection_id, "0").with_suffix(".json")
    pending = json.loads(record_path.read_bytes())
    recovered_manifest = RawStore(tmp_path).recover(collection_id)
    assert recovered_manifest["pages"] == [pending]
    assert recovered_manifest["next_cursor"] == "next"
    assert recovered_manifest["complete"] is False
    assert store.recover(collection_id) == recovered_manifest
    recovered = RawStore(tmp_path).write_page(
        collection_id, "0", {"rows": [1]}, next_cursor="next", row_count=1
    )
    assert recovered["received_at"] == pending["received_at"]
    assert store.manifest(collection_id)["row_count"] == 1
    assert not list((tmp_path / collection_id).rglob(".tmp-*"))


def test_failed_raw_publication_recovered_without_overwriting_metadata(tmp_path, monkeypatch):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    link = storage.os.link

    def fail_raw(source, target):
        if str(target).endswith(".bin"):
            raise OSError("raw failure")
        return link(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "link", fail_raw)
        with pytest.raises(OSError, match="raw failure"):
            store.write_page(collection_id, "0", {"value": 1}, row_count=1)
    assert not _raw_path(tmp_path, collection_id, "0").exists()
    before = store.manifest(collection_id)
    with pytest.raises(CorruptStoreError, match="Pending raw page is missing"):
        store.recover(collection_id)
    assert store.manifest(collection_id) == before
    with pytest.raises(ImmutablePageError):
        store.write_page(collection_id, "0", {"value": 2}, row_count=1)
    store.write_page(collection_id, "0", {"value": 1}, row_count=1)
    assert store.manifest(collection_id)["page_count"] == 1


def test_recover_completed_page_preserves_snapshot_and_retries_failed_commit(tmp_path, monkeypatch):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    payload = {"captured_at": "2026-01-03T10:00:00Z", "rows": [1]}
    metadata = {"recorded_at": "2026-01-03T10:01:00Z"}
    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(OSError, match="disk"):
            store.write_page(
                collection_id, "0", payload, row_count=1, metadata=metadata, complete=True
            )
        before = store.manifest(collection_id)
        with pytest.raises(OSError, match="disk"):
            store.recover(collection_id)
        assert store.manifest(collection_id) == before
    raw_path = _raw_path(tmp_path, collection_id, "0")
    pending = json.loads(raw_path.with_suffix(".json").read_bytes())
    recovered = store.recover(collection_id)
    assert recovered["pages"] == [pending]
    assert recovered["complete"] is True
    assert recovered["row_count"] == 1
    assert recovered["next_cursor"] is None
    assert json.loads(store.read_page(collection_id, "0")) == payload
    assert recovered["pages"][0]["metadata"] == metadata
    assert store.recover(collection_id) == recovered


@pytest.mark.parametrize("corruption", ["raw", "journal_name", "sequence", "extra_journal"])
def test_recover_rejects_corrupt_or_ambiguous_pending_without_manifest_changes(
    tmp_path, monkeypatch, corruption
):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(OSError, match="disk"):
            store.write_page(collection_id, "0", {"value": 1}, row_count=1)
    raw_path = _raw_path(tmp_path, collection_id, "0")
    record_path = raw_path.with_suffix(".json")
    if corruption == "raw":
        raw_path.write_bytes(b"corrupt")
    elif corruption == "journal_name":
        record_path.rename(record_path.with_name("wrong.json"))
    elif corruption == "sequence":
        record = json.loads(record_path.read_bytes())
        record["sequence"] = 2
        record_path.write_text(json.dumps(record))
    else:
        record_path.with_name("extra.json").write_bytes(record_path.read_bytes())
    before = store.manifest(collection_id)
    with pytest.raises(CorruptStoreError):
        store.recover(collection_id)
    assert store.manifest(collection_id) == before


def test_cyclic_request_and_page_rejected_without_partial_writes(tmp_path):
    store = RawStore(tmp_path)
    request_data = {}
    request_data["self"] = request_data
    with pytest.raises(RawStoreError, match="valid JSON"):
        store.begin("synthetic", "daily", request_data)
    assert list(tmp_path.iterdir()) == []
    collection_id = _collection(store)
    page = []
    page.append(page)
    before = store.manifest(collection_id)
    with pytest.raises(RawStoreError, match="valid JSON"):
        store.write_page(collection_id, "0", page)
    assert store.manifest(collection_id) == before
    assert list((tmp_path / collection_id / "pages").iterdir()) == []


def test_corrupt_raw_or_manifest_reference_rejected(tmp_path):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    store.write_page(collection_id, "0", {"value": 1}, row_count=1)
    _raw_path(tmp_path, collection_id, "0").write_bytes(b"corrupt")
    with pytest.raises(CorruptStoreError, match="checksum"):
        store.read_page(collection_id, "0")
    with pytest.raises(CorruptStoreError, match="checksum"):
        store.manifest(collection_id)

    other_id = store.begin("synthetic", "different", {})
    store.write_page(other_id, "0", {}, row_count=0)
    path = tmp_path / other_id / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["pages"][0]["page_id"] = "unknown"
    path.write_text(json.dumps(manifest))
    with pytest.raises(CorruptStoreError, match="missing"):
        store.manifest(other_id)


@pytest.mark.parametrize("page_id", ["../escape", "..", "/abs", "a/b", "a\\b"])
def test_invalid_page_paths_rejected(tmp_path, page_id):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    with pytest.raises(RawStoreError, match="page_id"):
        store.write_page(collection_id, page_id, {})
    with pytest.raises(RawStoreError, match="collection_id"):
        store.manifest("../escape")


def test_unknown_pages_and_symlinks_rejected(tmp_path):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    with pytest.raises(KeyError, match="Unknown page_id"):
        store.read_page(collection_id, "unknown")
    other_id = "0" * 64
    (tmp_path / other_id).symlink_to(tmp_path / collection_id, target_is_directory=True)
    with pytest.raises(CorruptStoreError, match="symlink"):
        store.manifest(other_id)


@pytest.mark.parametrize(
    "request_data",
    [
        {"api_key": "sensitive"},
        {"params": {"LSE_API_KEY": "sensitive"}},
        {"headers": {"Authorization": "sensitive"}},
        {"password": "sensitive"},
        {"token": "sensitive"},
        {"url": "https://example.test/data?api_key=sensitive"},
        {"url": "https://user:password@example.test/data"},
    ],
)
def test_credentials_rejected_before_any_manifest_write(tmp_path, request_data):
    store = RawStore(tmp_path)
    with pytest.raises(RawStoreError, match="Credentials"):
        store.begin("synthetic", "daily", request_data)
    assert list(tmp_path.iterdir()) == []


def test_threads_append_and_duplicate_replay_preserve_all_pages(tmp_path):
    store = RawStore(tmp_path)
    collection_id = _collection(store)

    def write(index):
        return RawStore(tmp_path).write_page(
            collection_id, str(index), {"rows": [index]}, row_count=1
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, list(range(12)) * 2))
    manifest = store.manifest(collection_id)
    assert manifest["page_count"] == manifest["row_count"] == 12
    assert {page["page_id"] for page in manifest["pages"]} == {str(i) for i in range(12)}


def test_processes_use_same_collection_lock(tmp_path):
    store = RawStore(tmp_path)
    collection_id = _collection(store)
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_process_write, args=(tmp_path, collection_id, str(i)))
        for i in range(3)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    assert store.manifest(collection_id)["page_count"] == 3
    assert store.manifest(collection_id)["row_count"] == 3
