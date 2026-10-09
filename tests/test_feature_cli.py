"""Exercise the documented local-file command and protect existing exports."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

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
