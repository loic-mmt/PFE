"""Downloader guards and date boundaries using a fake SDK; no credentials/network."""

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

        def economics(self, symbol, **options):
            calls.append((symbol, options))
            return [dict(date="2025-01-01", value=2), dict(date="2025-01-02", value=3)]

        def economic_calendar(self, **options):
            calls.append(("calendar", options))
            return [
                dict(id=1, datetime="2025-01-01T14:00:00Z"),
                dict(id=2, datetime="2025-01-02T00:00:00Z"),
            ]

    fake_module = types.ModuleType("lse")
    fake_module.LSE = FakeLSE
    monkeypatch.setitem(sys.modules, "lse", fake_module)
    path = Path(__file__).resolve().parents[1] / "scripts/download_macro.py"
    spec = importlib.util.spec_from_file_location("download_macro", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("LSE_API_KEY", "test-not-a-real-key")
    output = tmp_path / "snapshot.json"
    args = [
        "download_macro",
        "--series",
        "fdtr",
        "--calendar",
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


def test_download_caps_rows_filters_exclusive_end_and_records_limits(downloader):
    module, calls, output, _ = downloader
    module.main()
    payload = json.loads(output.read_text())
    assert len(payload["series"]["fdtr"]) == 1
    assert len(payload["calendar"]) == 1
    assert payload["limit_reached"] == {"fdtr": True, "calendar": True}
    assert all(options["limit"] == 2 for _, options in calls)
    assert "key" not in output.read_text()


def test_no_overwrite_or_network_when_output_exists(downloader):
    module, calls, output, _ = downloader
    output.write_text("keep")
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
    assert output.read_text() == "keep"


def test_oversize_never_writes_file(downloader):
    module, _, output, args = downloader
    args.extend(["--max-bytes", "10"])
    with pytest.raises(ValueError, match="max-bytes"):
        module.main()
    assert not output.exists()


def test_provider_exceeding_requested_rows_rejected(downloader):
    module, _, output, args = downloader
    args[args.index("--limit") + 1] = "1"
    with pytest.raises(ValueError, match="more rows"):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize(
    "option,value",
    [("--limit", "5001"), ("--max-bytes", "0"), ("--end", "bad"), ("--start", "2025-01-03")],
)
def test_invalid_arguments_fail_before_network(downloader, option, value):
    module, calls, _, args = downloader
    if option in args:
        args[args.index(option) + 1] = value
    else:
        args.extend([option, value])
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
