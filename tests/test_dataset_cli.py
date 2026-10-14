"""Local-only dataset and P3 collection commands."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from pfe.datasets import load_model_dataset


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_cli_local_reproducible(tmp_path):
    dates = pd.date_range("2026-01-01T22:00:00Z", periods=12, freq="D")
    points = [
        dict(
            entity_id="A",
            observation_date=day.isoformat(),
            available_at=day.isoformat(),
            published_at=None,
            value=100 + i,
            metric="close",
            unit="USD",
            source_id="fixture",
            revision_id=str(i),
            schema_version="1",
        )
        for i, day in enumerate(dates)
    ]
    source, cfg = tmp_path / "source.parquet", tmp_path / "config.json"
    pd.DataFrame(points).to_parquet(source)
    cfg.write_text(
        json.dumps(
            dict(
                decisions=[d.isoformat() for d in dates],
                entity_ids=["A"],
                features=[dict(name="close", metric="close", unit="USD", source_id="fixture")],
                lookback=2,
                labels=dict(price_feature="close", horizon=1),
                split=dict(train_end="2026-01-06T00:00:00Z", validation_end="2026-01-09T00:00:00Z"),
                scale=True,
            )
        )
    )
    cli = script("build_dataset")
    args = [
        "--observations",
        str(source),
        "--config",
        str(cfg),
        "--output",
        str(tmp_path / "dataset"),
    ]
    first, second = cli.main(args), cli.main(args)
    assert first == second
    arrays, manifest = load_model_dataset(tmp_path / "dataset")
    assert arrays["sequence_x"].shape == (11, 2, 1)
    assert manifest["configuration"]["input_sha256"][str(source)]
    assert manifest["scaler"]["fit_scope"] == "unique_training_window_cells"


def test_cot_cli_offline_and_explicit_receipt(tmp_path):
    cfg, source = tmp_path / "config.json", tmp_path / "source.json"
    cfg.write_text(
        json.dumps(
            dict(
                report="legacy_futures",
                market_code="001602",
                entity_id="wheat",
                start="2026-01-01",
                end="2026-01-31",
            )
        )
    )
    source.write_text(
        '[{"cftc_contract_market_code":"001602","report_date_as_yyyy_mm_dd":"2026-01-06","open_interest_all":"100"}]'
    )
    cli = script("download_p3")
    args = [
        "--source",
        "cot",
        "--config",
        str(cfg),
        "--snapshot",
        str(source),
        "--snapshot-id",
        "fixture",
        "--store-dir",
        str(tmp_path / "raw"),
    ]
    with pytest.raises(SystemExit):
        cli.main(args)
    args += ["--received-at", "2026-01-09T20:30:00Z"]
    assert cli.main(args) == cli.main(args)


def test_estimate_cli_does_not_invent_a_proprietary_network_api(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    with pytest.raises(SystemExit):
        script("download_p3").main(
            [
                "--source",
                "estimates",
                "--config",
                str(cfg),
                "--run",
                "--snapshot-id",
                "fixture",
                "--store-dir",
                str(tmp_path / "raw"),
            ]
        )
