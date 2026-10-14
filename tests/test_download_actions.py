"""Offline CLI snapshots, immutable replay and append-only action revisions."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from pfe.data.qualification import DiscoveryResponse
from pfe.data.storage import RawStore


@pytest.fixture
def action_cli():
    path = Path(__file__).parents[1] / "scripts/download_actions.py"
    spec = importlib.util.spec_from_file_location("download_actions_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _files(tmp_path, ratio=2):
    fields = {
        "split": {
            "symbol": "ticker",
            "action_id": "id",
            "effective_date": "date",
            "ratio": "factor",
        },
        "dividend": {
            "symbol": "ticker",
            "action_id": "id",
            "effective_date": "date",
            "amount": "cash",
            "currency": "ccy",
        },
    }
    snapshot = {
        "split": [{"ticker": "SYN", "id": "s1", "date": "2026-01-06", "factor": ratio}],
        "dividend": [
            {"ticker": "SYN", "id": "d1", "date": "2026-01-20", "cash": 1.0, "ccy": "USD"}
        ],
    }
    mapping = tmp_path / "fields.json"
    snapshot_path = tmp_path / "snapshot.json"
    mapping.write_text(json.dumps(fields))
    snapshot_path.write_text(json.dumps(snapshot))
    return mapping, snapshot_path, snapshot


def _args(tmp_path, mapping, snapshot=None):
    mode = ["--snapshot", str(snapshot)] if snapshot else ["--run", "--snapshot-id", "online-one"]
    return [
        *mode,
        "--symbol",
        "SYN",
        "--entity-id",
        "company",
        "--instrument-id",
        "share",
        "--start",
        "2026-01-01",
        "--end",
        "2026-01-31",
        "--field-map",
        str(mapping),
        "--ratio-convention",
        "new_per_old",
        "--dividend-basis",
        "post_split_share",
        "--store-dir",
        str(tmp_path / "raw"),
        "--output",
        str(tmp_path / "actions.parquet"),
    ]


def test_offline_replay_keeps_availability_and_does_not_duplicate_revisions(action_cli, tmp_path):
    mapping, snapshot, _ = _files(tmp_path)
    args = _args(tmp_path, mapping, snapshot)
    collection_id = action_cli.main(args)
    first = pd.read_parquet(tmp_path / "actions.parquet")
    assert len(first) == 2
    assert first.published_at.isna().all()
    assert first.availability_basis.eq("receipt").all()
    assert first.point_in_time.eq(False).all()
    store = RawStore(tmp_path / "raw")
    original = store.manifest(collection_id)
    assert original["complete"] is True
    assert action_cli.main(args) == collection_id
    pd.testing.assert_frame_equal(first, pd.read_parquet(tmp_path / "actions.parquet"))
    assert store.manifest(collection_id) == original


def test_new_provider_revision_appends_without_replacing_old_split(action_cli, tmp_path):
    mapping, snapshot, source = _files(tmp_path)
    old_id = action_cli.main(_args(tmp_path, mapping, snapshot))
    source["split"][0]["factor"] = 3
    snapshot.write_text(json.dumps(source))
    new_id = action_cli.main(_args(tmp_path, mapping, snapshot))
    assert old_id != new_id
    history = pd.read_parquet(tmp_path / "actions.parquet")
    assert history.loc[history.action_type.eq("split"), "ratio"].tolist() == [2.0, 3.0]
    assert len(history.loc[history.action_type.eq("dividend")]) == 1
    store = RawStore(tmp_path / "raw")
    assert json.loads(store.read_page(old_id, "split"))["response"][0]["factor"] == 2


def test_same_revision_cannot_be_reinterpreted_with_opposite_ratio_convention(action_cli, tmp_path):
    mapping, snapshot, _ = _files(tmp_path)
    args = _args(tmp_path, mapping, snapshot)
    action_cli.main(args)
    before = (tmp_path / "actions.parquet").read_bytes()
    args[args.index("new_per_old")] = "old_per_new"
    with pytest.raises(SystemExit):
        action_cli.main(args)
    assert (tmp_path / "actions.parquet").read_bytes() == before


def test_full_cap_retained_as_incomplete_not_empty_or_absent(action_cli, tmp_path):
    mapping, snapshot, _ = _files(tmp_path)
    collection_id = action_cli.main([*_args(tmp_path, mapping, snapshot), "--limit", "1"])
    manifest = RawStore(tmp_path / "raw").manifest(collection_id)
    assert manifest["complete"] is False
    assert manifest["pages"][-1]["metadata"]["incomplete"] is True
    assert len(pd.read_parquet(tmp_path / "actions.parquet")) == 2


def test_online_transport_is_explicit_bounded_and_cached(action_cli, tmp_path, monkeypatch):
    mapping, _, source = _files(tmp_path)
    calls = []

    class FakeTransport:
        def __init__(self, key):
            assert key == "synthetic-secret"

        def get(self, endpoint, params, *, max_bytes):
            calls.append((endpoint, params, max_bytes))
            kind = "split" if endpoint.endswith("stock_splits") else "dividend"
            rows = source[kind]
            return DiscoveryResponse(rows, len(json.dumps(rows).encode()))

    monkeypatch.setenv("LSE_API_KEY", "synthetic-secret")
    monkeypatch.setattr(action_cli, "LSEDataTransport", FakeTransport)
    args = _args(tmp_path, mapping)
    collection_id = action_cli.main(args)
    assert [call[0] for call in calls] == ["/ref/stock_splits", "/ref/dividends"]
    assert all(call[1]["limit"] == 5000 and call[1]["symbol"] == "SYN" for call in calls)
    original_manifest = RawStore(tmp_path / "raw").manifest(collection_id)
    original_history = pd.read_parquet(tmp_path / "actions.parquet")
    monkeypatch.delenv("LSE_API_KEY")
    assert action_cli.main(args) == collection_id
    assert len(calls) == 2
    assert RawStore(tmp_path / "raw").manifest(collection_id) == original_manifest
    pd.testing.assert_frame_equal(original_history, pd.read_parquet(tmp_path / "actions.parquet"))
    assert "synthetic-secret" not in json.dumps(original_manifest)


def test_uncached_online_page_requires_key_and_remains_incomplete(
    action_cli, tmp_path, monkeypatch, capsys
):
    mapping, _, _ = _files(tmp_path)
    monkeypatch.delenv("LSE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        action_cli.main(_args(tmp_path, mapping))
    assert "LSE_API_KEY" in capsys.readouterr().err
    (manifest_path,) = (tmp_path / "raw").glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["complete"] is False
    assert manifest["page_count"] == 0
    assert not (tmp_path / "actions.parquet").exists()


def test_snapshot_cap_rejected_before_any_raw_collection(action_cli, tmp_path):
    mapping, snapshot, _ = _files(tmp_path)
    with pytest.raises(SystemExit):
        action_cli.main([*_args(tmp_path, mapping, snapshot), "--max-bytes", "8"])
    assert not (tmp_path / "raw").exists()


def test_empty_provider_response_never_proves_action_absence_and_can_append_later(
    action_cli, tmp_path
):
    mapping, snapshot, source = _files(tmp_path)
    snapshot.write_text(json.dumps({"split": [], "dividend": []}))
    collection_id = action_cli.main(_args(tmp_path, mapping, snapshot))
    assert pd.read_parquet(tmp_path / "actions.parquet").empty
    store = RawStore(tmp_path / "raw")
    summary = json.loads(store.read_page(collection_id, "summary"))
    assert summary["window_complete"] is False
    assert summary["provider_completeness"] == "unqualified"
    assert summary["absence_scope"] == "unqualified"
    assert (
        store.manifest(collection_id)["pages"][-1]["metadata"]["completion_scope"]
        == "local_normalization_only"
    )
    snapshot.write_text(json.dumps(source))
    action_cli.main(_args(tmp_path, mapping, snapshot))
    assert len(pd.read_parquet(tmp_path / "actions.parquet")) == 2
