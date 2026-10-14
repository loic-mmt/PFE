"""Native daily download integration, entirely synthetic provider responses."""

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from pfe.data.lse_transport import LSEDataTransport
from pfe.data.qualification import DiscoveryError, DiscoveryLimitError, DiscoveryResponse
from pfe.data.storage import RawStore


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_daily", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_rows():
    return [
        dict(
            ts=f"{day} 00:00:00",
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000,
            session_scope="rth",
            price_adjustment="total_return",
        )
        for day, price in (("2026-09-04", 100), ("2026-09-08", 102), ("2026-09-09", 103))
    ]


def _args(tmp_path, *extra):
    return [
        "--timeframe",
        "1d",
        "--start",
        "2026-09-04",
        "--end",
        "2026-09-10",
        "--calendar",
        "XNYS",
        "--daily-label",
        "session_date",
        "--dataset",
        "etf",
        "--data-dir",
        str(tmp_path),
        *extra,
    ]


@pytest.mark.parametrize(
    "tickers",
    [["--ticker", "QQQ,SPY"], ["--ticker", "QQQ", "--ticker", "SPY"], ["--tickers", "QQQ", "SPY"]],
)
def test_native_multi_ticker_receipts_unknown_scope_and_immutable_replay(
    downloader, native_rows, tmp_path, monkeypatch, tickers
):
    calls = []

    class FakeTransport:
        def __init__(self, key):
            assert key == "synthetic-test-key"

        def get(self, endpoint, params, *, max_bytes):
            calls.append((endpoint, params))
            assert params["timeframe"] == "1d" and params["limit"] == 5000
            return DiscoveryResponse(native_rows, len(json.dumps(native_rows).encode()))

    def no_bulk(*args, **kwargs):
        raise AssertionError("Daily used an unbounded SDK history export")

    monkeypatch.setattr(downloader, "LSEDataTransport", FakeTransport)
    monkeypatch.setattr(downloader, "LSE", no_bulk)
    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    args = _args(tmp_path, *tickers)
    ids = downloader.main(args)
    assert len(ids) == 2 and [params["symbol"] for _, params in calls] == ["QQQ", "SPY"]
    outputs = sorted((tmp_path / "processed").glob("*.parquet"))
    assert len(outputs) == 2
    first_bytes = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}
    store = RawStore(tmp_path / "raw" / "lse" / "snapshots", max_page_bytes=8_388_608)
    original = [store.manifest(collection_id) for collection_id in ids]
    for path, collection_id in zip(outputs, ids, strict=True):
        frame = pd.read_parquet(path)
        assert list(frame.session_date) == ["2026-09-04", "2026-09-08", "2026-09-09"]
        assert frame.session_scope.eq("unknown").all()
        assert frame.price_adjustment.eq("unknown").all()
        assert frame.entity_id.isna().all()
        assert not frame.point_in_time.any()
        assert str(frame.timestamp.dt.tz) == "UTC"
        assert frame.timestamp.iloc[0].hour == 13 and frame.timestamp.iloc[0].minute == 30
        saved = json.loads(store.read_page(collection_id, "window-000000"))
        assert saved["rows"][0]["ts"] == "2026-09-04 00:00:00"  # Source untouched.
        assert (frame.available_at >= pd.Timestamp(saved["captured_at"])).all()
        sidecar = json.loads(path.with_suffix(".manifest.json").read_text())
        assert sidecar["coverage"]["expected_sessions"] == 3
        assert sidecar["coverage"]["missing_sessions"] == []
        assert sidecar["historical_point_in_time"] == "unqualified"
        assert sidecar["completion_scope"] == "local_normalization_only"
    monkeypatch.delenv("LSE_API_KEY")
    monkeypatch.setattr(downloader, "LSEDataTransport", no_bulk)
    assert downloader.main(args) == ids
    assert [store.manifest(collection_id) for collection_id in ids] == original
    assert all(
        (path.read_bytes(), path.stat().st_mtime_ns) == first_bytes[path] for path in outputs
    )


def test_daily_endpoint_failure_resumes_saved_windows(
    downloader, native_rows, tmp_path, monkeypatch
):
    calls, fail = [], True

    class FakeTransport:
        def __init__(self, key):
            pass

        def get(self, endpoint, params, *, max_bytes):
            calls.append(params["start"])
            if params["start"] == "2026-09-06" and fail:
                raise DiscoveryError("Synthetic interrupted daily response")
            rows = [row for row in native_rows if params["start"] <= row["ts"][:10] < params["end"]]
            return DiscoveryResponse(rows, len(json.dumps(rows).encode()))

    monkeypatch.setattr(downloader, "LSEDataTransport", FakeTransport)
    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    args = _args(tmp_path, "--window-days", "2")
    with pytest.raises(SystemExit):
        downloader.main(args)
    fail = False
    ids = downloader.main(args)
    assert calls == ["2026-09-04", "2026-09-06", "2026-09-06", "2026-09-08"]
    store = RawStore(tmp_path / "raw" / "lse" / "snapshots", max_page_bytes=8_388_608)
    assert store.manifest(ids[0])["complete"]
    assert len(pd.read_parquet(next((tmp_path / "processed").glob("*.parquet")))) == 3


@pytest.mark.parametrize(
    "extra",
    [
        ["--calendar", "UNKNOWN_CALENDAR"],
        ["--currency", "usd"],
        ["--limit", "5001"],
        ["--latency", "-1s"],
    ],
)
def test_invalid_daily_declarations_stop_before_provider(downloader, tmp_path, monkeypatch, extra):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid declaration contacted provider")

    monkeypatch.setattr(downloader, "LSEDataTransport", forbidden)
    with pytest.raises(SystemExit):
        downloader.main(_args(tmp_path, *extra))


def test_row_cap_leaves_incomplete_raw_snapshot_without_output(
    downloader, native_rows, tmp_path, monkeypatch
):
    calls = []

    class FakeTransport:
        def __init__(self, key):
            pass

        def get(self, endpoint, params, *, max_bytes):
            calls.append(endpoint)
            rows = native_rows[:2]
            return DiscoveryResponse(rows, len(json.dumps(rows).encode()))

    monkeypatch.setattr(downloader, "LSEDataTransport", FakeTransport)
    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    args = _args(tmp_path, "--limit", "2")
    with pytest.raises(SystemExit):
        downloader.main(args)
    manifests = list((tmp_path / "raw" / "lse" / "snapshots").glob("*/manifest.json"))
    manifest = json.loads(manifests[0].read_text())
    assert not manifest["complete"]
    assert not list((tmp_path / "processed").glob("*.parquet"))
    with pytest.raises(SystemExit):
        downloader.main(args)
    assert len(calls) == 1


def test_transport_caps_before_decode_preserves_missing_volume_and_redacts_keys():
    calls = []

    class Opener:
        def open(self, request, *, timeout):
            calls.append(request)
            return io.BytesIO(
                json.dumps([{"ts": "2026-09-04 00:00:00", "echo": "test-key"}]).encode()
            )

    transport = LSEDataTransport("test-key", opener=Opener())
    response = transport.get(
        "/candles", {"symbol": "QQQ", "timeframe": "1d", "limit": 3}, max_bytes=500
    )
    assert response.payload == [{"ts": "2026-09-04 00:00:00", "echo": "[redacted]"}]
    assert "volume" not in response.payload[0]
    assert calls[0].get_method() == "GET" and calls[0].full_url.startswith(
        "https://api.londonstrategicedge.com/vault/candles?"
    )
    with pytest.raises(DiscoveryLimitError):
        transport.get("/candles", {"limit": 3}, max_bytes=2)
    with pytest.raises(DiscoveryError):
        transport.get("/export", {"limit": 3}, max_bytes=500)
    with pytest.raises(DiscoveryError):
        transport.get("/candles", {"limit": 3, "api_key": "test-key"}, max_bytes=500)
    assert len(calls) == 2


def test_transport_does_not_echo_http_error_body_or_credentials():
    class Opener:
        def open(self, request, *, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 403, "test-key", {}, io.BytesIO(b"test-key")
            )

    with pytest.raises(DiscoveryError) as error:
        LSEDataTransport("test-key", opener=Opener()).get(
            "/ref/dividends", {"symbol": "AAPL", "limit": 3}, max_bytes=500
        )
    assert str(error.value) == "LSE row HTTP 403"


def test_provisional_daily_remains_ineligible_after_close_and_replay(
    downloader, native_rows, tmp_path, monkeypatch
):
    from pfe.data.sessions import exchange_schedule
    from pfe.features.daily import compute_daily_features

    class FakeTransport:
        def __init__(self, key):
            pass

        def get(self, endpoint, params, *, max_bytes):
            rows = native_rows[:1]
            return DiscoveryResponse(rows, len(json.dumps(rows).encode()))

    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(downloader, "LSEDataTransport", FakeTransport)
    monkeypatch.setattr(downloader, "_receipt", lambda: "2026-09-04T18:00:00Z")
    args = _args(tmp_path, "--end", "2026-09-05")
    downloader.main(args)
    output = next((tmp_path / "processed").glob("*.parquet"))
    before = output.read_bytes()
    bars = pd.read_parquet(output)
    assert bars["quality.session_complete"].eq(False).all()
    assert not bars["quality.feature_eligible"].any()
    assert bars.ingested_at.iloc[0] < bars.session_close.iloc[0]
    features = compute_daily_features(
        bars,
        schedule=exchange_schedule("2026-09-04", "2026-09-04", calendar="XNYS"),
        names=["returns.simple"],
        decision_at="2026-09-10T00:00:00Z",
    )
    assert features["returns.simple"].isna().all()
    monkeypatch.setattr(downloader, "_receipt", lambda: "2026-10-12T12:00:00Z")
    monkeypatch.delenv("LSE_API_KEY")
    downloader.main(args)
    assert output.read_bytes() == before
    assert not pd.read_parquet(output)["quality.feature_eligible"].any()


def test_end_boundary_filtered_before_non_session_validation(
    downloader, native_rows, tmp_path, monkeypatch
):
    rows = [native_rows[0], {**native_rows[0], "ts": "2026-09-06 00:00:00"}]

    class FakeTransport:
        def __init__(self, key):
            pass

        def get(self, endpoint, params, *, max_bytes):
            return DiscoveryResponse(rows, len(json.dumps(rows).encode()))

    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(downloader, "LSEDataTransport", FakeTransport)
    downloader.main(_args(tmp_path, "--end", "2026-09-06"))
    bars = pd.read_parquet(next((tmp_path / "processed").glob("*.parquet")))
    assert list(bars.session_date) == ["2026-09-04"]


def test_sdk_native_utc_open_labels_are_explicitly_normalized(
    downloader, native_rows, tmp_path, monkeypatch
):
    rows = [{**row, "ts": row["ts"].replace("00:00:00", "13:30:00")} for row in native_rows]

    class FakeTransport:
        def __init__(self, key):
            pass

        def get(self, endpoint, params, *, max_bytes):
            return DiscoveryResponse(rows, len(json.dumps(rows).encode()))

    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(downloader, "LSEDataTransport", FakeTransport)
    downloader.main(_args(tmp_path, "--daily-label", "open"))
    bars = pd.read_parquet(next((tmp_path / "processed").glob("*.parquet")))
    assert len(bars) == 3 and (bars.timestamp == bars.session_open).all()


def test_legacy_parquet_snapshot_replays_without_key_or_provider(
    downloader, raw_bars, tmp_path, monkeypatch
):
    calls = []

    class FakeLSE:
        def __init__(self, *, api_key):
            assert api_key == "synthetic-test-key"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def history(self, symbol, **kwargs):
            calls.append(kwargs)
            assert kwargs["dataframe"] is False
            destination = Path(kwargs["dest"])
            destination.mkdir(parents=True)
            path = destination / "stocks_QQQ_1m.parquet"
            raw_bars.rename(columns={"timestamp": "ts"}).to_parquet(path)
            return str(path)

    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(downloader, "LSE", FakeLSE)
    args = ["--start", "2026-01-05", "--end", "2026-01-06", "--data-dir", str(tmp_path)]
    ids = downloader.main(args)
    output = next((tmp_path / "processed").glob("*.parquet"))
    previous = output.read_bytes()
    output.unlink()  # Recreate a missing export from immutable saved processed bytes.

    def no_provider(*args, **kwargs):
        raise AssertionError("Legacy replay contacted provider")

    monkeypatch.delenv("LSE_API_KEY")
    monkeypatch.setattr(downloader, "LSE", no_provider)
    assert downloader.main([*args, "--resume"]) == ids
    assert len(calls) == 1 and output.read_bytes() == previous
    summary = json.loads(output.with_suffix(".manifest.json").read_text())
    assert summary["coverage"]["byte_cap_scope"] == "after_sdk_transfer_before_parquet_loading"
