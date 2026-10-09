"""Check the download integration without a key or network request."""

import importlib.util
import sys
from pathlib import Path

import pytest

from pfe import load_bars


@pytest.fixture
def download_module():
    pytest.importorskip("lse")
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("ticker", "symbol"), [(None, "QQQ"), (" spy ", "SPY"), ("AAPL", "AAPL"), ("brk.b", "BRK.B")]
)
@pytest.mark.parametrize(
    ("start", "end", "export_start", "export_end", "expected_rows"),
    [
        ("2026-08-31", "2026-09-10", "2026-08-31", "2026-09-10", 2),
        (
            "2026-01-05T14:30:00Z",
            "2026-01-05T14:31:00Z",
            "2026-01-05",
            "2026-01-06",
            1,
        ),
        (
            "2026-01-05T15:30:00+01:00",
            "2026-01-05T15:31:00+01:00",
            "2026-01-05",
            "2026-01-06",
            1,
        ),
    ],
)
def test_download_uses_date_only_export_and_excludes_end(
    download_module,
    raw_bars,
    tmp_path,
    monkeypatch,
    start,
    end,
    export_start,
    export_end,
    expected_rows,
    ticker,
    symbol,
):
    if start == "2026-08-31":
        raw_bars["timestamp"] = ["2026-08-31T14:30:00Z", "2026-09-09T14:31:00Z"]
    calls = []

    class FakeLSE:
        def __init__(self, *, api_key):
            assert api_key == "synthetic-test-key"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def history(self, symbol, **kwargs):
            assert kwargs["start"] == export_start
            assert kwargs["end"] == export_end
            calls.append((symbol, kwargs))
            Path(kwargs["dest"]).mkdir(parents=True)
            return raw_bars.rename(columns={"timestamp": "ts"})

    monkeypatch.setattr(download_module, "LSE", FakeLSE)
    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_data.py",
            *([] if ticker is None else ["--ticker", ticker]),
            "--start",
            start,
            "--end",
            end,
            "--data-dir",
            str(tmp_path),
        ],
    )
    download_module.main()
    assert calls[0][0] == symbol
    assert calls[0][1]["timeframe"] == "1m"
    assert Path(calls[0][1]["dest"]).parent.parent.name == symbol
    outputs = list((tmp_path / "processed").glob("*.parquet"))
    assert len(outputs) == 1
    assert outputs[0].name.startswith(f"{symbol}_1m_")
    assert len(load_bars(outputs[0])) == expected_rows
    with pytest.raises(SystemExit):
        download_module.main()
    assert len(calls) == 1  # Refuse to overwrite before contacting LSE.


def test_missing_key_stops_before_download(download_module, monkeypatch):
    monkeypatch.delenv("LSE_API_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["download_data.py", "--start", "2026-01-05", "--end", "2026-01-06"]
    )
    with pytest.raises(SystemExit):
        download_module.main()


@pytest.mark.parametrize("ticker", ["", " ", "../SPY", "SPY/QQQ", "SPY\\QQQ", ".", ".."])
def test_invalid_ticker_stops_before_download(download_module, monkeypatch, ticker):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid ticker reached the provider")

    monkeypatch.setattr(download_module, "LSE", forbidden)
    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_data.py", "--ticker", ticker, "--start", "2026-01-05", "--end", "2026-01-06"],
    )
    with pytest.raises(SystemExit) as exc:
        download_module.main()
    assert exc.value.code == 2


def test_different_tickers_can_share_date_window(download_module, raw_bars, tmp_path, monkeypatch):
    class FakeLSE:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def history(self, symbol, **kwargs):
            Path(kwargs["dest"]).mkdir(parents=True)
            return raw_bars

    monkeypatch.setattr(download_module, "LSE", FakeLSE)
    monkeypatch.setenv("LSE_API_KEY", "synthetic-test-key")
    for ticker in ("QQQ", "SPY"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "download_data.py",
                "--ticker",
                ticker,
                "--start",
                "2026-01-05",
                "--end",
                "2026-01-06",
                "--data-dir",
                str(tmp_path),
            ],
        )
        download_module.main()
    assert len(list((tmp_path / "processed").glob("*.parquet"))) == 2
