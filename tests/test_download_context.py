"""Offline CLI and injected online transports; public sources are never fetched."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pfe.data.storage import RawStore


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_context.py"
    spec = importlib.util.spec_from_file_location("download_context", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _files(tmp_path, config, body, suffix):
    cfg, source = tmp_path / "config.json", tmp_path / ("source" + suffix)
    cfg.write_text(json.dumps(config))
    source.write_bytes(body)
    output = tmp_path / "observations.parquet"
    args = [
        "--config",
        str(cfg),
        "--snapshot",
        str(source),
        "--output",
        str(output),
        "--store-dir",
        str(tmp_path / "raw"),
        "--received-at",
        "2026-09-03T12:00:00Z",
    ]
    return args, output


def test_eia_offline_no_key_network_idempotence_and_nullable_publication(
    cli, tmp_path, monkeypatch
):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    payload = {
        "response": {
            "frequency": "daily",
            "total": "1",
            "data": [{"period": "2026-09-01", "value": "71", "value-units": "Dollars per Barrel"}],
        },
        "request": {"params": {"api_key": "synthetic-offline-key"}},
    }
    args, output = _files(
        tmp_path,
        {"route": "petroleum/pri/spt", "entity_id": "declared_wti"},
        json.dumps(payload).encode(),
        ".json",
    )
    collection_id = cli.main(["--source", "eia", *args])
    before, mtime = output.read_bytes(), output.stat().st_mtime_ns
    frame = pd.read_parquet(output)
    assert frame.published_at.isna().all() and not frame.point_in_time.any()
    assert frame.available_at.iloc[0] == pd.Timestamp("2026-09-03T12:00:00Z")
    assert cli.main(["--source", "eia", *args]) == collection_id
    assert output.read_bytes() == before and output.stat().st_mtime_ns == mtime
    for path in (tmp_path / "raw").rglob("*"):
        if path.is_file():
            assert b"synthetic-offline-key" not in path.read_bytes()


def test_cboe_offline_source_receipt_and_licence_metadata(cli, tmp_path):
    body = b"DATE,OPEN,HIGH,LOW,CLOSE\n09/01/2026,16,18,15,17\n"
    args, output = _files(tmp_path, {"metrics": ["CLOSE"]}, body, ".csv")
    collection_id = cli.main(["--source", "cboe", *args])
    assert pd.read_parquet(output).observation_date.iloc[0] == "2026-09-01"
    store = RawStore(tmp_path / "raw")
    assert store.read_page(collection_id, "raw-file") == body
    assert "license_notes" in store.manifest(collection_id)["source_metadata"]


def test_pink_sheet_offline_xlsx_selected_schema(cli, tmp_path):
    excel = pytest.importorskip("openpyxl")
    book = excel.Workbook()
    book.active.title = "ObservedMonthly"
    book.active.append(["Month", "Copper"])
    book.active.append(["2026M08", 9600])
    data = io.BytesIO()
    book.save(data)
    book.close()
    config = dict(
        sheet_name="ObservedMonthly",
        header_row=0,
        period_column="Month",
        period_format="YYYYMmm",
        column_specs={"Copper": {"entity_id": "copper", "unit": "USD/ton", "currency": "USD"}},
    )
    args, output = _files(tmp_path, config, data.getvalue(), ".xlsx")
    collection_id = cli.main(["--source", "world_bank", *args])
    frame = pd.read_parquet(output)
    assert frame.frequency.iloc[0] == "monthly" and frame.unit.iloc[0] == "USD/ton"
    assert cli.main(["--source", "world_bank", *args]) == collection_id


def test_incomplete_eia_snapshot_never_marks_complete_output(cli, tmp_path):
    payload = {
        "response": {
            "frequency": "daily",
            "total": "2",
            "data": [{"period": "2026-09-01", "value": "71"}],
        }
    }
    args, output = _files(
        tmp_path, {"route": "energy", "entity_id": "factor"}, json.dumps(payload).encode(), ".json"
    )
    collection_id = cli.main(["--source", "eia", *args])
    assert not output.exists()
    partial = list(tmp_path.glob("observations.partial-*.parquet"))
    assert len(partial) == 1
    assert not json.loads(partial[0].with_suffix(".manifest.json").read_text())["complete"]
    assert not RawStore(tmp_path / "raw").manifest(collection_id)["complete"]


def test_online_is_explicit_and_injected_transport_replays_without_key(cli, tmp_path, monkeypatch):
    import pfe.data.http as http

    calls = []

    class FakeHTTP:
        def __init__(self, **kwargs):
            pass

        def get(self, url, *, params=None, max_bytes, headers=None):
            calls.append(url)
            assert params["api_key"] == "synthetic-api-key"
            payload = {
                "response": {
                    "frequency": "daily",
                    "total": "1",
                    "data": [{"period": "2026-09-01", "value": "71"}],
                }
            }
            return SimpleNamespace(
                body=json.dumps(payload).encode(), received_at="2026-09-03T12:00:00Z"
            )

    config, output = tmp_path / "config.json", tmp_path / "observations.parquet"
    config.write_text(json.dumps({"route": "energy", "entity_id": "factor"}))
    monkeypatch.setattr(http, "BoundedHTTPTransport", FakeHTTP)
    monkeypatch.setenv("EIA_API_KEY", "synthetic-api-key")
    args = [
        "--source",
        "eia",
        "--run",
        "--snapshot-id",
        "explicit",
        "--config",
        str(config),
        "--output",
        str(output),
        "--store-dir",
        str(tmp_path / "raw"),
    ]
    collection_id = cli.main(args)
    monkeypatch.delenv("EIA_API_KEY")
    assert cli.main(args) == collection_id and len(calls) == 1


@pytest.mark.parametrize("extra", [[], ["--run"], ["--snapshot", "missing.csv"]])
def test_cli_modes_and_receipts_required_before_output(cli, tmp_path, extra):
    config, output = tmp_path / "config.json", tmp_path / "out.parquet"
    config.write_text("{}")
    with pytest.raises(SystemExit):
        cli.main(["--source", "cboe", "--config", str(config), "--output", str(output), *extra])
    assert not output.exists()
