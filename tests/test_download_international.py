"""Offline CLI receipts, immutable files and bounded feature exports."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/download_international.py"
    spec = importlib.util.spec_from_file_location("download_international", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path, *, features=False):
    config, snapshot, output = (
        tmp_path / name for name in ("config.json", "source.html", "out.parquet")
    )
    config.write_text(json.dumps({"year": 2026, "entity_id": "tsmc"}))
    snapshot.write_text(
        "<h2>2026 Monthly Revenue</h2><p>Millions of New Taiwan Dollars</p>"
        "<table><tr><td>Jan.</td><td>100</td><td>20%</td></tr></table>"
    )
    result = [
        "--source",
        "tsmc",
        "--config",
        str(config),
        "--snapshot",
        str(snapshot),
        "--received-at",
        "2026-02-10T08:00:00Z",
        "--store-dir",
        str(tmp_path / "raw"),
        "--output",
        str(output),
    ]
    if features:
        feature_config = tmp_path / "features.json"
        feature_config.write_text(
            json.dumps(
                {
                    "decisions": ["2026-02-10T07:59:59Z", "2026-02-10T08:00:00Z"],
                    "series": [
                        {
                            "name": "tsmc",
                            "entity_id": "tsmc",
                            "metric": "revenue",
                            "unit": "million_TWD",
                            "source_id": "tsmc",
                            "frequency": "monthly",
                        }
                    ],
                }
            )
        )
        result.extend(
            [
                "--features-config",
                str(feature_config),
                "--features-output",
                str(tmp_path / "features.parquet"),
            ]
        )
    return result


def test_cli_source_feature_export_and_idempotent_replay(cli, tmp_path):
    command = args(tmp_path, features=True)
    first = cli.main(command)
    second = cli.main(command)
    assert first == second
    observations = pd.read_parquet(tmp_path / "out.parquet")
    assert len(observations) == 2
    output = pd.read_parquet(tmp_path / "features.parquet")
    assert pd.isna(output.iloc[0]["international.tsmc.level"])
    assert output.iloc[1]["international.tsmc.level"] == 100
    assert json.loads((tmp_path / "out.manifest.json").read_text())["complete"]


def test_cli_explicit_receipt_required(cli, tmp_path):
    command = args(tmp_path)
    index = command.index("--received-at")
    del command[index : index + 2]
    with pytest.raises(SystemExit):
        cli.main(command)
    assert not (tmp_path / "out.parquet").exists()


def test_cli_output_immutable(cli, tmp_path):
    command = args(tmp_path)
    (tmp_path / "out.parquet").write_bytes(b"previous content")
    with pytest.raises(SystemExit):
        cli.main(command)
    assert (tmp_path / "out.parquet").read_bytes() == b"previous content"


def test_cli_cap_rejects_input(cli, tmp_path):
    command = args(tmp_path) + ["--max-response-bytes", "20"]
    with pytest.raises(SystemExit):
        cli.main(command)
    assert not (tmp_path / "out.parquet").exists()


def test_cli_online_requires_snapshot_id_before_network(cli, tmp_path):
    command = args(tmp_path)
    index = command.index("--snapshot")
    del command[index : index + 2]
    command.append("--run")
    with pytest.raises(SystemExit):
        cli.main(command)


def test_cli_gdelt_partial_output_named_explicitly(cli, tmp_path):
    command = args(tmp_path)
    command[command.index("--source") + 1] = "gdelt"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "entity_id": "news",
                "query": "semiconductors",
                "start": "2026-02-09T00:00:00Z",
                "end": "2026-02-10T00:00:00Z",
                "max_records": 1,
            }
        )
    )
    (tmp_path / "source.html").write_text(
        json.dumps({"articles": [{"url": "https://example.org/a", "seendate": "20260209T120000Z"}]})
    )
    cli.main(command)
    assert (tmp_path / "out.partial.parquet").exists()
    assert not (tmp_path / "out.parquet").exists()
    assert not json.loads((tmp_path / "out.partial.manifest.json").read_text())["complete"]
