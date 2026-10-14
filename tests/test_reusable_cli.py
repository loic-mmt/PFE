"""Local CLI replay, literal credential loading and bounded dataset artifacts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfe import TimeframeSpec, build_timeframe_observations, iter_saved_partitions


def script(name):
    path = Path(__file__).parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_literals_do_not_execute_or_overwrite(tmp_path, monkeypatch):
    module = script("verify_integrations")
    for key in module.KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FRED_API_KEY", "already_set")
    path = tmp_path / ".env"
    path.write_text(
        "export FRED_API_KEY=ignored\nEIA_API_KEY='$(touch never_created)'\n"
        'ESTAT_APP_ID="literal" # comment\nUNRELATED=ignored\n'
    )
    module.load_environment(path)
    import os

    assert os.environ["FRED_API_KEY"] == "already_set"
    assert os.environ["EIA_API_KEY"] == "$(touch never_created)"
    assert os.environ["ESTAT_APP_ID"] == "literal"
    assert not (tmp_path / "never_created").exists()
    path.write_text("x" * 65_537)
    with pytest.raises(ValueError, match="budget"):
        module.load_environment(path)


def test_partition_cli_runs_and_replays_without_network(tmp_path):
    times = pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="min")
    values = 100 + np.arange(20)
    bars = pd.DataFrame(
        dict(
            open=values,
            high=values + 1,
            low=values - 1,
            close=values,
            volume=1.0,
            available_at=times + pd.Timedelta("1min"),
        ),
        index=times,
    )
    features = build_timeframe_observations(
        {"A": bars},
        [TimeframeSpec("1min", ("close",), {"close": "USD"})],
        source_frequency="1min",
        source_id="fixture",
        revision_id="1",
    )
    path = tmp_path / "observations.parquet"
    features.observations.sort_values("available_at").to_parquet(path, row_group_size=4)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                decisions=[t.isoformat() for t in times + pd.Timedelta("1min")],
                entity_ids=["A"],
                features=features.registry.to_dict()["features"],
                lookback=3,
                labels=dict(price_feature="tf[1min].close", horizon=2),
                scale=True,
                train_end="2026-01-01T00:15:00Z",
            )
        )
    )
    module = script("build_partitioned_dataset")
    args = [
        "--observations",
        str(path),
        "--config",
        str(config),
        "--output",
        str(tmp_path / "out"),
        "--partition-size",
        "5",
        "--batch-rows",
        "3",
    ]
    first = module.main(args)
    second = module.main(args)
    assert first == second and len(first["shards"]) == 4
    shards = list(iter_saved_partitions(tmp_path / "out"))
    assert sum(len(arrays["sequence_x"]) for arrays, _ in shards) == 18
    assert all("sequence_x_scaled" in arrays for arrays, _ in shards)


def test_pipeline_cli_is_local_by_default(tmp_path):
    path = tmp_path / "input.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                steps=[
                    dict(
                        name="local",
                        operation="local_tables",
                        parameters=dict(paths={"table": str(path)}),
                        outputs={"table": "metadata/local.parquet"},
                    )
                ]
            )
        )
    )
    result = script("run_pipeline").main(
        ["--config", str(config), "--output", str(tmp_path / "out")]
    )
    assert result["requests"] == 0 and result["manifest"]["complete"]
