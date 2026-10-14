"""Exercise the documented local-file command and protect existing exports."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pfe import normalize_bars


def test_cli_writes_parquet_and_catalog_and_refuses_overwrite(raw_bars, tmp_path):
    bars = normalize_bars(raw_bars)
    source = tmp_path / "input.parquet"
    output = tmp_path / "features.parquet"
    catalog = tmp_path / "catalog.csv"
    bars.to_parquet(source)
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_features.py"
    command = [
        sys.executable,
        str(script),
        "--input",
        str(source),
        "--output",
        str(output),
        "--catalog",
        str(catalog),
        "--frequency",
        "2min",
        "--origin",
        bars.index[0].isoformat(),
        "--windows",
        "2",
        "--groups",
        "returns",
        "volatility",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "Saved 1 bars" in result.stdout
    frame = pd.read_parquet(output)
    assert len(frame) == 1
    assert "path.realized_variance" in frame
    assert "returns.simple" in frame
    assert str(frame.available_at.dt.tz) == "UTC"
    assert list(pd.read_csv(catalog).name) == list(frame.columns)
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0
    assert "Output already exists" in refused.stderr
    assert len(pd.read_parquet(output)) == 1


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_daily_cli_uses_sessions_preserves_availability_and_writes_catalog(tmp_path, format):
    from pfe.data.daily import normalize_daily_bars
    from pfe.data.sessions import exchange_schedule

    schedule = exchange_schedule("2026-09-04", "2026-09-09", calendar="XNYS")
    raw = pd.DataFrame(
        {
            "timestamp": ["2026-09-04", "2026-09-08", "2026-09-09"],
            "open": [100, 102, 103],
            "high": [101, 103, 104],
            "low": [99, 101, 102],
            "close": [100, 102, 103],
            "volume": [1000, 1100, 1200],
        }
    )
    bars = normalize_daily_bars(
        raw,
        schedule=schedule,
        instrument_id="001",
        source_id="synthetic",
        ingested_at="2026-09-10T12:00:00Z",
    )
    source, output, catalog = [
        tmp_path / name for name in (f"daily.{format}", "features.parquet", "catalog.csv")
    ]
    if format == "csv":
        bars.to_csv(source, index=False)
    else:
        bars.to_parquet(source)
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_features.py"
    command = [
        sys.executable,
        str(script),
        "--daily",
        "--input",
        str(source),
        "--output",
        str(output),
        "--catalog",
        str(catalog),
        "--calendar",
        "XNYS",
        "--windows",
        "2",
        "--groups",
        "returns",
        "volatility",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "Saved 3 daily bars" in completed.stdout
    frame = pd.read_parquet(output)
    assert list(frame.session_date) == ["2026-09-04", "2026-09-08", "2026-09-09"]
    assert frame["returns.simple"].iloc[1] == pytest.approx(0.02)
    assert frame.available_at.equals(bars.available_at)
    assert str(frame.timestamp.dt.tz) == "UTC"
    assert frame.instrument_id.eq("001").all()
    assert not any(name.startswith("path.") for name in frame)
    inventory = pd.read_csv(catalog)
    assert list(inventory.name) == list(frame)
    assert inventory.availability.eq("explicit available_at").all()
    previous = output.read_bytes()
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0 and "Output already exists" in refused.stderr
    assert output.read_bytes() == previous


@pytest.mark.parametrize("extra", [["--frequency", "24h"], ["--batch-size", "2"], ["--path-only"]])
def test_daily_cli_rejects_intraday_options_before_writing(tmp_path, extra):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_features.py"
    output = tmp_path / "features.parquet"
    command = [
        sys.executable,
        str(script),
        "--daily",
        "--input",
        str(tmp_path / "missing.parquet"),
        "--output",
        str(output),
        "--calendar",
        "XNYS",
        *extra,
    ]
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0
    assert not output.exists()
