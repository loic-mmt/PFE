"""Bounded options downloader, using a fake SDK and no real credentials/network."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def downloader(monkeypatch, tmp_path):
    calls = []

    class FakeLSE:
        def __init__(self, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def options(self, ticker, **options):
            calls.append(("chain", ticker, options))
            return [dict(ticker="QQQ251219C00100000"), dict(ticker="QQQ251219P00100000")]

        def options_flow(self, ticker, **options):
            calls.append(("flow", ticker, options))
            return [dict(ts="2025-01-01T12:00:00Z"), dict(ts="2025-01-02T00:00:00Z")]

        def option_candles(self, ticker, **options):
            calls.append(("candles", ticker, options))
            return [dict(minute="2025-01-01T12:00:00Z"), dict(minute="2025-01-02T00:00:00Z")]

    fake = types.ModuleType("lse")
    fake.LSE = FakeLSE
    monkeypatch.setitem(sys.modules, "lse", fake)
    path = Path(__file__).resolve().parents[1] / "scripts/download_options.py"
    spec = importlib.util.spec_from_file_location("download_options", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("LSE_API_KEY", "fake-test-key")
    output = tmp_path / "sample.json"
    args = [
        "download_options",
        "--ticker",
        "qqq",
        "--chain",
        "--flow",
        "--contract",
        "QQQ251219C00100000",
        "--start",
        "2025-01-01",
        "--end",
        "2025-01-02",
        "--limit",
        "2",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    return module, calls, output, args


def test_bounded_calls_end_exclusion_and_cap_metadata(downloader):
    module, calls, output, _ = downloader
    module.main()
    payload = json.loads(output.read_text())
    assert payload["underlying"] == "QQQ"
    assert len(payload["chain"]) == 2
    assert len(payload["flow"]) == 1
    assert len(payload["candles"]) == 1
    assert all(payload["limit_reached"].values())
    assert len(calls) == 3
    assert all(options["limit"] == 2 for _, _, options in calls)
    assert "fake-test-key" not in output.read_text()


def test_refuses_existing_output_before_network(downloader):
    module, calls, output, _ = downloader
    output.write_text("keep")
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
    assert output.read_text() == "keep"


def test_byte_cap_and_provider_row_cap(downloader):
    module, _, output, args = downloader
    args.extend(["--max-bytes", "10"])
    with pytest.raises(ValueError, match="max-bytes"):
        module.main()
    assert not output.exists()
    args[-1] = "1048576"
    args[args.index("--limit") + 1] = "1"
    with pytest.raises(ValueError, match="oversized"):
        module.main()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--ticker", "../bad"),
        ("--contract", "SPY251219C00100000"),
        ("--limit", "0"),
        ("--max-bytes", "0"),
        ("--end", "bad"),
        ("--start", "2025-01-03"),
        ("--expiry", "2025-99-01"),
    ],
)
def test_invalid_arguments_before_network(downloader, option, value):
    module, calls, _, args = downloader
    if option in args:
        args[args.index(option) + 1] = value
    else:
        args.extend([option, value])
    with pytest.raises(SystemExit):
        module.main()
    assert not calls


def test_no_selection_and_incomplete_strike_bounds(downloader):
    module, calls, _, args = downloader
    args.extend(["--strike-min", "100"])
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
