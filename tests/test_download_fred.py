"""Offline native API fixtures and safe cached online replay."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from pfe.data.http import HTTPResponse
from pfe.data.storage import RawStore


@pytest.fixture
def fred_cli():
    spec = importlib.util.spec_from_file_location(
        "download_fred_test", Path(__file__).parents[1] / "scripts/download_fred.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot():
    meta = {
        "id": "SYN",
        "title": "Synthetic rate",
        "units": "Percent",
        "frequency": "Daily",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "last_updated": "2026-04-01 09:00:00-05",
    }
    rows = [
        {
            "date": "2026-01-01",
            "value": "1.5",
            "realtime_start": "2026-04-01",
            "realtime_end": "2026-04-01",
        }
    ]

    def page(key, data):
        return {key: data, "count": len(data), "offset": 0, "limit": 1000}

    payloads = {
        "/series": {"seriess": [meta]},
        "/series/release": {"releases": [{"id": 82, "name": "Synthetic release"}]},
        "/series/observations": page("observations", rows),
        "/series/vintagedates": page("vintage_dates", ["2026-04-01"]),
        "/release/dates": page("release_dates", [{"release_id": 82, "date": "2026-04-01"}]),
    }
    return {
        "responses": [
            dict(
                endpoint=endpoint,
                payload=payload,
                received_at="2026-04-01T16:00:00Z",
                **({"release_id": 82} if endpoint == "/release/dates" else {}),
            )
            for endpoint, payload in payloads.items()
        ]
    }


def _args(tmp_path, snapshot=None):
    mode = ["--snapshot", str(snapshot)] if snapshot else ["--run", "--snapshot-id", "online-one"]
    return [
        *mode,
        "--series-id",
        "SYN",
        "--start",
        "2026-01-01",
        "--end",
        "2026-01-31",
        "--realtime-start",
        "2026-04-01",
        "--realtime-end",
        "2026-04-01",
        "--store-dir",
        str(tmp_path / "raw"),
        "--output",
        str(tmp_path / "observations.parquet"),
    ]


def test_offline_receipt_replay_preserves_values_and_availability(fred_cli, tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(_snapshot()))
    args = _args(tmp_path, snapshot)
    collection_id = fred_cli.main(args)
    frame = pd.read_parquet(tmp_path / "observations.parquet")
    assert frame.value.tolist() == [1.5]
    assert frame.available_at.iloc[0] == pd.Timestamp("2026-04-01T16:00:00Z")
    assert frame.published_at.isna().all()
    assert frame.point_in_time.eq(False).all()
    store = RawStore(tmp_path / "raw")
    manifest = store.manifest(collection_id)
    assert manifest["complete"] is True
    assert fred_cli.main(args) == collection_id
    assert store.manifest(collection_id) == manifest
    pd.testing.assert_frame_equal(frame, pd.read_parquet(tmp_path / "observations.parquet"))


def test_new_snapshot_appends_revision_without_overwriting_original(fred_cli, tmp_path):
    snapshot = tmp_path / "snapshot.json"
    source = _snapshot()
    snapshot.write_text(json.dumps(source))
    first_id = fred_cli.main(_args(tmp_path, snapshot))
    source["responses"][2]["payload"]["observations"][0]["value"] = "1.6"
    # Receipt of a distinct corrected latest snapshot must also be distinct.
    for response in source["responses"]:
        response["received_at"] = "2026-04-02T16:00:00Z"
    snapshot.write_text(json.dumps(source))
    second_id = fred_cli.main(_args(tmp_path, snapshot))
    assert first_id != second_id
    frame = pd.read_parquet(tmp_path / "observations.parquet")
    assert frame.value.tolist() == [1.5, 1.6]
    assert len(frame.revision_id.unique()) == 2


def test_date_level_offline_history_never_uses_last_updated_as_release(fred_cli, tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(_snapshot()))
    args = [
        *_args(tmp_path, snapshot),
        "--availability",
        "vintage_next_day",
        "--vintage-timezone",
        "America/New_York",
    ]
    fred_cli.main(args)
    frame = pd.read_parquet(tmp_path / "observations.parquet")
    assert frame.available_at.iloc[0] == pd.Timestamp("2026-04-02T04:00:00Z")
    assert frame.availability_basis.iloc[0] == "reconstructed"
    assert frame.published_at.isna().all()


def test_online_replay_needs_no_key_or_new_network_call(fred_cli, tmp_path, monkeypatch):
    source = _snapshot()
    calls = []

    class FakeTransport:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, *, params, max_bytes):
            endpoint = url.removeprefix("https://api.stlouisfed.org/fred")
            calls.append(endpoint)
            assert params["api_key"] == "synthetic-secret"
            entry = next(item for item in source["responses"] if item["endpoint"] == endpoint)
            return HTTPResponse(json.dumps(entry["payload"]).encode(), entry["received_at"])

    monkeypatch.setenv("FRED_API_KEY", "synthetic-secret")
    monkeypatch.setattr(fred_cli, "BoundedHTTPTransport", FakeTransport)
    args = _args(tmp_path)
    collection_id = fred_cli.main(args)
    store = RawStore(tmp_path / "raw")
    first = store.manifest(collection_id)
    monkeypatch.delenv("FRED_API_KEY")
    assert fred_cli.main(args) == collection_id
    assert len(calls) == 5
    assert store.manifest(collection_id) == first
    assert "synthetic-secret" not in json.dumps(first)


def test_missing_online_key_reports_only_required_credential(
    fred_cli, tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        fred_cli.main(_args(tmp_path))
    assert "FRED_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "observations.parquet").exists()


def test_offline_input_cap_rejected_before_raw_creation(fred_cli, tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(_snapshot()))
    with pytest.raises(SystemExit):
        fred_cli.main([*_args(tmp_path, snapshot), "--max-bytes", "8"])
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(
    "field,replacement",
    [("frequency", "Monthly"), ("seasonal_adjustment", "Seasonally Adjusted")],
)
def test_repeated_value_revision_cannot_silently_hide_changed_conventions(
    fred_cli, tmp_path, field, replacement
):
    snapshot = tmp_path / "snapshot.json"
    source = _snapshot()
    snapshot.write_text(json.dumps(source))
    args = _args(tmp_path, snapshot)
    fred_cli.main(args)
    output = tmp_path / "observations.parquet"
    original = output.read_bytes()
    source["responses"][0]["payload"]["seriess"][0][field] = replacement
    for response in source["responses"]:
        response["received_at"] = "2026-04-02T16:00:00Z"
    snapshot.write_text(json.dumps(source))
    with pytest.raises(SystemExit):
        fred_cli.main(args)
    assert output.read_bytes() == original
